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
import matplotlib.pyplot as plt

from tre_tvmc.utils import mpi_logmeanexp_jax
from ._vjp_chunked import vjp_chunked

def expect_2distr(
    log_pdf_new: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    log_pdf_old: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    log_expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    log_weights_fun: Callable,
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
        log_weights_fun,
        pars_new,
        pars_old,
        σ_new,
        σ_old,
        args,
        args_in_axes,
    )

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5, 11))
def _expect_2distr(
    n_chains,
    chunk_size,
    log_pdf_new,
    log_pdf_old,
    log_expected_fun,
    log_weights_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    args,
    args_in_axes,
):
    log_w = nkjax.apply_chunked(log_weights_fun, in_axes=(None, None, 0, 0, *args_in_axes), chunk_size=chunk_size)(pars_new, pars_old, σ_new, σ_old, *args)
    log_Floc = nkjax.apply_chunked(log_expected_fun, in_axes=(None, None, 0, 0, *args_in_axes), chunk_size=chunk_size)(pars_new, pars_old, σ_new, σ_old, *args)
    if n_chains is not None:
        log_Floc = log_Floc.reshape((n_chains, -1))
        log_w = log_w.reshape((n_chains, -1))
    log_B = compute_log_norm(log_w, n_chains=n_chains) # computes the parallel stats itself
    log_wc = log_w - log_B
    # NOTE: wc is defined such that its MEAN is 1, not its sum (!!!)
    wc = jnp.exp(log_wc)
    
    log_Lloc = log_Floc + log_wc # weight them here explicitly
    Lloc = jnp.exp(log_Lloc)
    
    L_stat = mpi_statistics(Lloc)
    
    aux = (L_stat, wc)
    return L_stat.mean, aux


def _print_locals(x, y):
    x = x.reshape(-1)
    plt.title("local info L")
    plt.scatter(np.arange(x.size), np.array(x))
    plt.show()

    y = y.reshape(-1)
    plt.title("local info w")
    plt.scatter(np.arange(y.size), np.array(y))
    plt.show()
    return None

def _expect_fwd_fid(
    n_chains,
    chunk_size,
    log_pdf_new,
    log_pdf_old,
    log_expected_fun,
    log_weights_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    args,
    args_in_axes,
):
    assert σ_new.ndim == 2
    
    # step 0: compute the log_w, so we can use them below for stability and simplicity
    log_w = nkjax.apply_chunked(log_weights_fun, in_axes=(None, None, 0, 0, *args_in_axes), chunk_size=chunk_size)(pars_new, pars_old, σ_new, σ_old, *args)    
    log_B = compute_log_norm(log_w).real # computes the parallel stats itself
    log_wc = log_w - log_B
    wc = jnp.exp(log_wc)
    # jax.debug.print("B = {B}", B=jnp.exp(log_B))
    # jax.debug.print("wc = {wc}", wc=jnp.mean(wc))
    
    # now: vjp to get Floc, \nabla Floc
    def _L_fun(pars_new, pars_old, σ_new, σ_old, log_wc, *all_args):
        # expected fun is no longer weighted
        log_Floc = log_expected_fun(pars_new, pars_old, σ_new, σ_old, *all_args)
        Lloc = jnp.exp(log_Floc + log_wc) # hopefully a more safe weighting
        return Lloc
        
    chunk_argnums = [2, 3, 4]
    chunk_argnums += [i+5 for i, ax in enumerate(args_in_axes) if ax is not None] 
    primals = [pars_new, pars_old, σ_new, σ_old, log_wc, *args]
    nondiff_argnums = np.arange(1, len(primals)).tolist()
    # very dumb thing of netket: returns a function that returns what vjp does...
    # SO: we have to do the vjp here already to get the forward...       
    vjp_out = vjp_chunked(
        _L_fun, 
        *primals, # primals
        chunk_size=chunk_size, 
        chunk_argnums=tuple(chunk_argnums),
        nondiff_argnums=tuple(nondiff_argnums),
        has_aux=False,
        return_forward=True,
    )
    
    n_samples = σ_new.shape[0]
    average_v = jnp.full((n_samples,), 1/n_samples)
    Lloc, grad_L = vjp_out(average_v) # Lloc = Aloc*wc

    if n_chains is not None:
        Lloc_r = Lloc.reshape((n_chains, -1))
    else:
        Lloc_r = Lloc
    L_stat = mpi_statistics(Lloc_r)
    L = L_stat.mean
    # Use the baseline trick to reduce the variance
    ΔLloc = Lloc - wc*L # this looks strange, since Lloc already contains wc
    
    # if mpi.n_nodes == 1:
    #     jax.debug.callback(_print_locals, safe_log(Lloc), log_wc)

    aux = (L_stat, wc)
    return (L, aux), (pars_new, pars_old, σ_new, σ_old, args, ΔLloc, grad_L, average_v)

def _expect_bwd_fid(n_chains, chunk_size, log_pdf_new, log_pdf_old, log_expected_fun, log_weights_fun, args_in_axes, residuals, dout):
    pars_new, pars_old, σ_new, σ_old, args, ΔLloc, grad_L, average_v = residuals
    dL, L_stat = dout
        
    def _logp_dL_fun(pars_new, pars_old, σ_new, σ_old, ΔLloc, *args):
        log_p = log_pdf_new(pars_new, σ_new) + log_pdf_old(pars_old, σ_old)
        # ΔLloc already contains the weights wc
        out = jax.vmap(jnp.multiply)(ΔLloc, log_p)
        return out
    
    chunk_argnums = [2, 3, 4]
    chunk_argnums += [5 + i for i, a in enumerate(args_in_axes) if a is not None]
    primals = [pars_new, pars_old, σ_new, σ_old, ΔLloc, *args]
    nondiff_argnums = np.arange(1, len(primals)).tolist()
        
    vjp_out = vjp_chunked(
        _logp_dL_fun,
        *primals, # primals
        chunk_size=chunk_size,
        chunk_argnums=tuple(chunk_argnums),
        nondiff_argnums=tuple(nondiff_argnums),
        has_aux=False,
        return_forward=False,
    )
    logp_dL_vjp_fun = vjp_out

    # netket is a bit annoying here, since it automatically chunks the cotangent
    
    grad_logp_dL = logp_dL_vjp_fun(average_v*dL)
    
    # grad_L = L_vjpfun(average_v)
    # move the mpi_mean to the external functions
    # jax.debug.print("a, b = {a},\n {b}", a=grad_logp_dL, b=grad_L)
    grad_f = trees_add(grad_logp_dL, grad_L)
    
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


@partial(jax.jit, static_argnums=(1,))
def compute_log_norm(log_w, n_chains=None):
    if n_chains is not None:
        log_w = log_w.reshape(n_chains, -1)
    return mpi_logmeanexp_jax(log_w)

# @jax.jit
# def compute_normalized_weights(log_w, log_N):
#     log_wc = log_w - log_N
#     return log_wc

@jax.jit
def trees_add(tree1, tree2):
    return jax.tree_util.tree_map(lambda x, y: x+y, tree1, tree2)

@jax.jit
def trees_diff(tree1, tree2):
    return jax.tree_util.tree_map(lambda x, y: x-y, tree1, tree2)


@jax.jit
def tree_scalar_multiply(scalar, tree):
    return jax.tree_util.tree_map(lambda x: scalar*x, tree)
