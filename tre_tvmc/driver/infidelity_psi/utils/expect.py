from typing import Callable, Tuple
from functools import partial
import jax.numpy as jnp
import jax
from netket.utils.types import PyTree
import netket.jax as nkjax
from netket.stats import statistics as mpi_statistics, Stats, total_size
import netket.jax as nkjax
import numpy as np
from netket.utils import mpi

from tre_tvmc.utils import mpi_logmeanexp_jax, safe_log

def expect_2distr(
    log_pdf_new: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    log_pdf_old: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    log_expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars_new: PyTree,
    pars_old: PyTree,
    σ_new: jnp.ndarray,
    σ_old: jnp.ndarray,
    args: PyTree,
    args_in_axes: PyTree,
    n_chains: int = None,
    chunk_size: int = None,
) -> Tuple[jnp.ndarray, Stats]:
    """
    Computes the expectation value over a log-pdf.

    Args:
        log_pdf:
        expected_ffun
    """

    return _expect_2distr(
        n_chains,
        chunk_size,
        log_pdf_new,
        log_pdf_old,
        log_expected_fun,
        pars_new,
        pars_old,
        σ_new,
        σ_old,
        args,
        args_in_axes,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 10))
def _expect_2distr(
    n_chains,
    chunk_size,
    log_pdf_new,
    log_pdf_old,
    log_expected_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    args,
    args_in_axes,
):
    log_L_σ, aux = nkjax.apply_chunked(partial(log_expected_fun, return_aux=True), in_axes=(None, None, 0, 0, *args_in_axes), chunk_size=chunk_size)(pars_new, pars_old, σ_new, σ_old, *args)
    if n_chains is not None:
        log_L_σ = log_L_σ.reshape((n_chains, -1))

    log_w = aux.get("log_w", None)
    if log_w is not None:
        log_N = compute_log_norm(log_w) # computes the parallel stats itself
        # WATCH OUT: NOW ALSO DOES THE WEIGHTING OF THE MEAN!!!
        # DON'T REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        L̄_stat, _ = compute_weighted_stats(log_L_σ, log_w, log_N)
    else:
        L_σ = jnp.exp(log_L_σ)
        L̄_stat = mpi_statistics(L_σ.T)

    return L̄_stat.mean, L̄_stat


def _expect_fwd_fid(
    n_chains,
    chunk_size,
    log_pdf_new,
    log_pdf_old,
    log_expected_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    args,
    args_in_axes,
):
    log_L_σ, aux = nkjax.apply_chunked(partial(log_expected_fun, return_aux=True), in_axes=(None, None, 0, 0, *args_in_axes), chunk_size=chunk_size)(pars_new, pars_old, σ_new, σ_old, *args)
    if n_chains is not None:
        log_L_σ_r = log_L_σ.reshape((n_chains, -1))
    else:
        log_L_σ_r = log_L_σ

    log_w = aux.get("log_w", None)
    if log_w is not None:
        log_N = compute_log_norm(log_w) # computes the parallel stats itself
        # WATCH OUT: NOW ALSO DOES THE WEIGHTING OF THE MEAN!!!
        # DON'T REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        L̄_stat, L_σ_r = compute_weighted_stats(log_L_σ_r, log_w, log_N)
    else:
        L_σ_r = jnp.exp(log_L_σ_r)
        L̄_stat = mpi_statistics(L_σ_r.T)
        log_N = 0

    L̄_σ = L̄_stat.mean

    # Use the baseline trick to reduce the variance
    L_σ = L_σ_r.reshape(-1)
    ΔL_σ = L_σ - L̄_σ

    return (L̄_σ, L̄_stat), (pars_new, pars_old, σ_new, σ_old, args, ΔL_σ, log_N)

def _expect_bwd_fid(n_chains, chunk_size, log_pdf_new, log_pdf_old, log_expected_fun, args_in_axes, residuals, dout):
    pars_new, pars_old, σ_new, σ_old, args, ΔL_σ, log_N = residuals
    dL̄, dL̄_stats = dout
    
    # print("->_expect_bwd_fid")
    # we don't actually need to do this, since it doesn't depend on the parameters (!!!!)
    # log_p_old = nkjax.apply_chunked(log_pdf_old, in_axes=(None, 0), chunk_size=chunk_size)(pars_old, σ_old)

    def f(pars_new, pars_old, σ_new, σ_old, ΔL_σ, *all_args):
        # let's do this inside so the gradients are 0
        log_p = log_pdf_new(pars_new, σ_new) + log_pdf_old(pars_old, σ_old)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        # the following part should actually be skipped if we take things holomorphic (!!!)
        # we don't return aux here
        log_term2 = log_expected_fun(pars_new, pars_old, σ_new, σ_old, *all_args)
        term2 = jnp.exp(log_term2 - log_N) # only that term needs recorrecting, the other has been corrected
        out = term1 + term2 # has length of the chunk size
        # we move the mean outside to the vjp part, which includes a sum
        return out
    
    # chunk_argnums = [2, 3, 4, 5]
    # chunk_argnums += [i+6 for i, ax in enumerate(args_in_axes) if ax is not None] 
    chunk_argnums = [2, 3, 4]
    chunk_argnums += [i+5 for i, ax in enumerate(args_in_axes) if ax is not None] 
    # let's never differentiate through the argnums
    
    # primals = [pars_new, pars_old, σ_new, σ_old, ΔL_σ, log_p_old, *args]
    primals = [pars_new, pars_old, σ_new, σ_old, ΔL_σ, *args]
    nondiff_argnums = np.arange(1, len(primals)).tolist()
        
    pb = nkjax.vjp_chunked(
        f, 
        *primals, # primals
        chunk_size=chunk_size, 
        chunk_argnums=tuple(chunk_argnums),
        nondiff_argnums=tuple(nondiff_argnums),
        has_aux=False
    )

    # netket is a bit annoying here, since it automatically chunks the cotangent
    # n_samples = log_p_old.shape[0]
    n_samples = ΔL_σ.shape[0]
    dL̄ = jnp.repeat(dL̄ / n_samples, n_samples) # correct the mean factor !
    grad_f = pb(dL̄)
    # move the mpi_mean to the external functions
    
    # grad_f = tuple(primals)
    # grad_f = (
    #     *grad_f, # pars_new
    #     # let's just return the primals for the rest instead of doing efforts
    #     *primals[1:]
    # )
    
    # !!!!!!!!!!!!!!
    # IMPORTANT: MUST HAVE SAME STRUCTURE AS NON-DIFF ARGUMENTS OF CUSTOM FWD
    # FILL WITH INPUT TO GET THE SAME STRUCTURE!
    grad_f = (
        *grad_f,
        pars_old,
        σ_new,
        σ_old,
        args,
    )       

    return grad_f


_expect_2distr.defvjp(_expect_fwd_fid, _expect_bwd_fid)

# @jax.jit
# def mpi_covariance(x, y):
#     assert x.shape == y.shape
#     x_stats = mpi_statistics(x)
#     y_stats = mpi_statistics(y)
#     xy_stats = mpi_statistics(x*y)
#     cov = xy_stats.mean - x_stats.mean*y_stats.mean
#     return cov

@jax.jit
def compute_weighted_stats(log_Llocs, log_w, log_N):
    """ NOT CORRECT: ALREADY CONTAINS A WEIGHT!!!"""
    log_wc = log_w - log_N
    wc = jnp.exp(log_wc)
    wc = wc.reshape(*log_Llocs.shape)
    wc_stats = mpi_statistics(wc.T)
    
    # replace already everything to include the normalization
    log_Llocs -= log_N
    Llocs = jnp.exp(log_Llocs)
    Lstats = mpi_statistics(Llocs.T)
    
    # jax.debug.print("w = {w}", w=wc_stats)
    
    cov_F_wc = mpi_statistics(Llocs*wc).mean - Lstats.mean*wc_stats.mean # correct for mean?
    
    new_var_F = Lstats.variance + (Lstats.mean**2)*wc_stats.variance - 2*Lstats.mean*cov_F_wc
    
    # jax.debug.print("var contributions = {A}    |   {B}    |   {C}", A=Lstats.variance, B=(Lstats.mean**2)*var_wc, C = - 2*Lstats.mean*cov_F_wc)
    
    n_total = total_size(Llocs)
    Lstats = Lstats.replace(
        variance=new_var_F, 
        error_of_mean=jnp.sqrt(new_var_F/n_total)
    )
    return Lstats, Llocs

@jax.jit
def compute_log_norm(log_w):
    return mpi_logmeanexp_jax(log_w)
