from functools import partial
import jax
import jax.numpy as jnp
from netket.jax import logsumexp_cplx as logsumexp
import netket as nk
from netket import jax as nkjax
from netket.operator import ContinuousOperator
from netket.vqs import MCState, expect, expect_and_forces, get_local_kernel
from netket.utils import mpi
from typing import Callable, Optional, Any
from netket.utils.types import PyTree, Array, Union
from netket.stats import statistics, Stats
import warnings
from functools import partial
from netket.jax import vjp as nkvjp

from .operator import InfidelityOperatorUPsi
from ..utils import expect_2distr
from tre_tvmc.driver.utils import possibly_undo_chunk_size, mpi_logmeanexp_jax, safe_log
from tre_tvmc.driver.utils import print_mpi
from tre_tvmc.operator import IdentityOperator


def validate_arguments(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: int = None):
    """ Function to check the restrictions we imposed in this file. """
    if not isinstance(op._U, ContinuousOperator):
        raise ValueError("Only works with continuous operators.")
    if not isinstance(op._U_dagger, ContinuousOperator):
        raise ValueError("Only works with continuous operators.")

@partial(jax.jit, static_argnums=(0,3))
def _log_Upsi(apply_fun: Callable, pars: PyTree, σ: Array, O: ContinuousOperator, *op_args: Optional[PyTree]):
    # O: _expect_kernel computes [OPsi]/Psi, but we need Log(Opsi)
    # so we compute
    # Log(Opsi) = Log([OPsi]/Psi) + Log(Psi)
    # for example for an interaction potential we get
    # Log[VPsi] = Log[V] + Log[Psi], which is great
    logpsi_σ = apply_fun(pars, σ)
    logUlocal_σ = safe_log(O._expect_kernel(apply_fun, pars, σ, op_args))
    logUpsi_σ = logUlocal_σ + logpsi_σ
    return logUpsi_σ

def get_local_kernel_arguments_infidelity_with_in_axes(vstate: MCState, target: MCState, O: ContinuousOperator, O_dagger: ContinuousOperator):
    # THIS IS FOR THE OVERALL THING (!!!)
    σ = vstate.samples
    σ_t = target.samples
    args = (O._pack_arguments(), O_dagger._pack_arguments())
    args_in_axes = (None, None)
    return σ, σ_t, args, args_in_axes

###

# WE DO IT WITH MACHINE_POW=1!!!
# def _apply_fun_from_sqrt(apply_fun_sqrt):
#     def _apply_fun(params, x):
#         return 2*apply_fun_sqrt(params, x)
#     return _apply_fun

def _local_kernel_log_weights_jax(
        O: ContinuousOperator,
        O_dagger: ContinuousOperator,
        apply_fun_psi: Callable,
        apply_fun_phi: Callable,
        pars_psi: PyTree,
        pars_phi: PyTree,
        x: Array,
        y: Array,
        O_args: Any,
        O_dagger_args: Any,
        include_R: bool = True,
        include_sqrt: bool = True,
    ):
    # make proper functions
    # apply_fun_psi = _apply_fun_from_sqrt(apply_fun_psi_sqrt)
    # apply_fun_phi = _apply_fun_from_sqrt(apply_fun_phi_sqrt)

    log_w = 0

    # x samples
    log_psi_x = apply_fun_psi(pars_psi, x)
    # y samples
    log_phi_y = apply_fun_phi(pars_phi, y)
    
    # compute the weights
    if include_sqrt:
        log_w +=  log_phi_y.real + log_psi_x.real
    
    if include_R:
        log_Uphi_y = _log_Upsi(apply_fun_phi, pars_phi, y, O, *O_args)
        log_Rloc = log_Uphi_y - log_phi_y
        log_w += 2*log_Rloc.real    
    
    return log_w

def _local_kernel_log_fidelity_jax( # NO weights
        O: ContinuousOperator,
        O_dagger: ContinuousOperator,
        apply_fun_psi: Callable,
        apply_fun_phi: Callable,
        pars_psi: PyTree,
        pars_phi: PyTree,
        x: Array,
        y: Array,
        O_args: Any,
        O_dagger_args: Any,
        cv_coeff: float = None,
    ):
            
    # make proper functions
    # apply_fun_psi = _apply_fun_from_sqrt(apply_fun_psi_sqrt)
    # apply_fun_phi = _apply_fun_from_sqrt(apply_fun_phi_sqrt)
    
    # x samples
    log_Uphi_x = _log_Upsi(apply_fun_phi, pars_phi, x, O, *O_args)
    log_psi_x = apply_fun_psi(pars_psi, x)
    # y samples
    log_psi_y = apply_fun_psi(pars_psi, y)
    log_Uphi_y = _log_Upsi(apply_fun_phi, pars_phi, y, O, *O_args)
    
    log_Floc_xy = (log_Uphi_x - log_psi_x) + (log_psi_y - log_Uphi_y)
    
    log_res = log_Floc_xy
    if cv_coeff is not None:
        log_res = logsumexp(
            jnp.stack([
                log_res,
                2*log_Floc_xy.real,
                jnp.zeros_like(log_res)
            ], axis=-1),
            b=jnp.array([1, cv_coeff, -cv_coeff]),
            axis=-1
        )
                        
    return log_res


@expect.dispatch
def infidelity_chunked(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: int):
    L_stats, wc = _infidelity_and_grad_chunked(vstate, op, chunk_size, return_grad=False)
    op._wc = wc
    return L_stats

@expect_and_forces.dispatch
def infidelity_and_forces_chunked(  # noqa: F811
    vstate: MCState,
    op: InfidelityOperatorUPsi,
    chunk_size: int,
    *,
    mutable,
):
    L_stats, L_grad, wc = _infidelity_and_grad_chunked(vstate, op, chunk_size, return_grad=True)
    op._wc = wc
    return L_stats, L_grad

def _infidelity_and_grad_chunked(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: int, return_grad: bool=True):
    # now in a single function
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    validate_arguments(vstate, op, chunk_size=chunk_size)
    
    sigma, sigma_t, args, args_in_axes = get_local_kernel_arguments_infidelity_with_in_axes(vstate, op.target, op._U, op._U_dagger)
    
    # functions to compute Floc etc
    local_log_infidelity_kernel = nk.utils.HashablePartial(_local_kernel_log_fidelity_jax, op._U, op._U_dagger)
    local_log_weights_kernel = nk.utils.HashablePartial(_local_kernel_log_weights_jax, op._U, op._U_dagger, include_sqrt=op._sample_sqrt)
   
    return infidelity_sampling_MCState(
        chunk_size,
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        local_log_infidelity_kernel,
        local_log_weights_kernel,
        sigma,
        sigma_t,
        args,
        args_in_axes,
        op.cv_coeff,
        return_grad=return_grad,
    )

def local_log_weights_kernel_wrapper( # add some spice
        local_log_weights_kernel,
        afun, params, σ, 
        afun_t, params_t, σ_t, 
        args,
        model_state=None, 
        model_state_t=None,
    ):
    """ Part handling the local fidelity kernel. """
    
    # to be sure, we block some gradients (speeds some things up)
    
    model_state = model_state or {}
    model_state_t = model_state_t or {}
    
    W = {"params": params, **model_state}
    W_t = {"params": params_t, **model_state_t}
    
    batch_shape = σ.shape[:-1]
    σ = σ.reshape(-1, σ.shape[-1])
    σ_t = σ_t.reshape(-1, σ_t.shape[-1])

    log_w = local_log_weights_kernel(afun, afun_t, W, W_t, σ, σ_t, *args)
    log_w = log_w.reshape(*batch_shape)            
    return log_w

def local_log_fidelity_kernel_wrapper( # add some spice
        local_log_fidelity_kernel,
        afun, params, σ, 
        afun_t, params_t, σ_t, 
        args,
        cv_coeff=None, 
        model_state=None, 
        model_state_t=None,
    ):
    """ Part handling the local fidelity kernel. """
    
    # to be sure, we block some gradients (speeds some things up)
    
    model_state = model_state or {}
    model_state_t = model_state_t or {}
    
    W = {"params": params, **model_state}
    W_t = {"params": params_t, **model_state_t}
    
    batch_shape = σ.shape[:-1]
    σ = σ.reshape(-1, σ.shape[-1])
    σ_t = σ_t.reshape(-1, σ_t.shape[-1])

    log_res = local_log_fidelity_kernel(afun, afun_t, W, W_t, σ, σ_t, *args, cv_coeff=cv_coeff)
    log_res = log_res.reshape(*batch_shape)
            
    return log_res

@partial(jax.jit, static_argnames=("afun", "afun_t", "return_grad", "chunk_size", "local_braket_fun", "local_weights_fun", "args_in_axes",))
def infidelity_sampling_MCState(
    chunk_size,
    afun,
    afun_t,
    params,
    params_t,
    model_state,
    model_state_t,
    local_braket_fun,
    local_weights_fun,
    sigma,
    sigma_t,
    args,
    args_in_axes,
    cv_coeff,
    return_grad,
):
    N = sigma.shape[-1]
    
    if sigma.ndim == 3:
        n_chains = sigma.shape[-2]
    else:
        n_chains = None

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)
    
    
    def expect_kernel(params):

        def kernel_fun(params, params_t, σ, σ_t, *args):
            return local_log_fidelity_kernel_wrapper(
                local_braket_fun,
                afun, params, σ,
                afun_t, params_t, σ_t,
                args,
                cv_coeff=cv_coeff, 
                model_state=model_state, 
                model_state_t=model_state_t,
            )

        # TRYING SOMETHING NEW: PASS p, NOT Q
        log_pdf = lambda params, σ: 2 * afun({"params": params, **model_state}, σ).real
        log_pdf_t = (
            lambda params, σ: 2 * afun_t({"params": params, **model_state_t}, σ).real
        )
        
        log_weights_fn = lambda params, params_t, σ, σ_t, *args : local_log_weights_kernel_wrapper(
            local_weights_fun,
            afun, params, σ,
            afun_t, params_t, σ_t,
            args,
            model_state=model_state, 
            model_state_t=model_state_t,
        )
        
        return expect_2distr(
            log_pdf,
            log_pdf_t,
            kernel_fun,
            log_weights_fn,
            params,
            params_t,
            σ,
            σ_t,
            args,
            args_in_axes,
            n_chains=n_chains,
            chunk_size=chunk_size,
        )

    if not return_grad:
        F, aux = expect_kernel(params)
        F_stats, wc = aux
        I_stats = F_stats.replace(mean=1 - F)
        return I_stats, wc

    out = nkvjp(
        expect_kernel, 
        params,
        has_aux=True, 
        conjugate=True,
    )
    F, F_vjp_fun, aux = out # (primals, vjp_fun, aux)
    F_stats, wc = aux
    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)
    I_grad = jax.tree_map(lambda x: -x, F_grad)
    I_stats = F_stats.replace(mean=1 - F)

    return I_stats, I_grad, wc
