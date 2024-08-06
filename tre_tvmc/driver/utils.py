import netket as nk
import numpy as np
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
import functools
from typing import Sequence, Optional
import netket.jax as nkjax 
from netket.utils import mpi
import tqdm
import matplotlib.pyplot as plt
from flax.traverse_util import flatten_dict
import json
import os, sys
import time


# def device_count():
#     return sharding.device_count()

def device_count():
    return mpi.n_nodes

def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False

def print_mpi(*args, **kwargs):
    if mpi.node_number == 0:
        print(*args, **kwargs)

def tqdm_mpi(x):
    if mpi.node_number == 0:
        return tqdm.tqdm(x)
    else:
        return x
    
def copy_frozen_dict(frozen_dict):
    return type(frozen_dict)({**frozen_dict})

def copy_variational_state(vs, n_hot=0, copy_samples=True, **kwargs):
    # warning: does not use the initial sampler state values
    apply_fun = kwargs.get("apply_fun", None)
    init_fun = kwargs.get("init_fun", None)
    variables = kwargs.get("variables", None)
    if variables is not None:
        variables = copy_frozen_dict(variables)
    # model = vs.model
    # apply = model.apply if apply_fun is None else apply_fun
    # init = model.init if init_fun is None else init_fun
    apply = vs._apply_fun if apply_fun is None else apply_fun
    init = vs._init_fun if init_fun is None else init_fun

    if isinstance(vs, nk.vqs.MCState) or hasattr(vs, "samples"):
        copy_vs = nk.vqs.MCState(
            vs.sampler,
            apply_fun=apply,
            init_fun=init,
            n_samples=vs.n_samples,
            # model=model,
            chunk_size=vs.chunk_size,
            n_discard_per_chain=vs.n_discard_per_chain,
            variables=variables,
        )
        if variables is None:
            copy_vs.variables = copy_frozen_dict(vs.variables)
        vs.samples # initialize it already
        if copy_samples and hasattr(vs.sampler_state, "σ"):
            copy_vs.sampler_state = copy_vs.sampler_state.replace(σ=vs.sampler_state.σ)
        for _ in range(n_hot):
            copy_vs.reset()
            copy_vs.samples
        return copy_vs
    elif isinstance(vs, nk.vqs.FullSumState) or hasattr(vs, "_all_states"):
        copy_vs = nk.vqs.FullSumState(
            vs.hilbert, 
            apply_fun=apply,
            init_fun=init,
            variables=variables,
            # model=model,
        )
        if variables is None:
            copy_vs.parameters = copy_frozen_dict(vs.parameters)
        return copy_vs
    else:
        raise NotImplementedError(f"cannot copy this kind of state {type(vs)}")
    

def calc_full_chunk_size(vstate):
    n_devices = device_count()
    if vstate.n_samples % n_devices != 0:
        raise ValueError("Cannot compute n_samples = {} on n_devices = {}".format(vstate.n_samples, n_devices))
    chunk_size = vstate.n_samples // n_devices
    return chunk_size

def possibly_undo_chunk_size(vstate, chunk_size):
    if chunk_size == vstate.n_samples // device_count():
        return None
    else:
        return chunk_size

def sparsify(U):
    return U.to_sparse()

def make_Ustate(vs, op=None, normalize=False):
    vec = vs.to_array(normalize=normalize)
    if op is None:
        return vec
    else:
        U_sp = sparsify(op)
        Ustate = U_sp @ vec
        if normalize:
            Ustate /= np.linalg.norm(Ustate)
        return Ustate

def default_inner_callback(*args):
    return True

class LargeHilbertPlotter:
    def __init__(self, n_samples=None, resample=True):
        self.n_samples = n_samples
        self._samples = None
        self._vs = None
        self.resample = resample
        
    def reset(self, vs=None):
        self._samples = None
        self._vs = vs
        
    @property
    def samples(self):
        if self._samples is None:
            x = self._vs.samples
            x = x.reshape(-1, x.shape[-1])
            if self.n_samples is not None and x.shape[0] > self.n_samples:
                x = x[:self.n_samples,:]
            self._samples = x
        return self._samples
        
    def __call__(self, step_nr, log_data, driver):
        if mpi.node_number != 0:
            return True
        state = driver.state
        if isinstance(state, nk.vqs.FullSumState):
            return True
        if self.resample or driver.step_count == 0:
            self.reset(vs=state)
        target = driver._op.target
        U = driver._op._U
        # make states in a limited hilbert space
        x = self.samples
        v1 = _state_on_samples(x, state, normalize=True)
        v2 = _state_on_samples(x, target, U=U, normalize=True)
        e = np.real(driver._loss_stats.mean)
        g = jnp.linalg.norm(jax.flatten_util.ravel_pytree(driver._loss_grad)[0])
        _plot_vec_and_close(v1, v2, modulus=False, match_phase=True, name=f"SAMPLED({driver.step_count})", e=e, g=g)
        return True

def _state_on_samples(x, state, U=None, normalize=False):
    assert x.ndim == 2
    if U is None:
        lv = state.log_value(x)
    else:
        xp, mels = U.get_conn_padded(x)
        xp = xp.reshape(-1, xp.shape[-1])
        lv = state.log_value(xp)
        lv = lv.reshape(*mels.shape)
        lv = nkjax.logsumexp_cplx(lv, b=mels, axis=-1)
    v = jnp.exp(lv)
    if normalize: # this means normalizaiton on the samples only (!!!)
        v /= jnp.linalg.norm(v)
    return v  

def plot_projected_states_callback(step_nr, log_data, driver):
    if mpi.n_nodes > 1:
        return True
    e = driver._loss_stats
    g = driver._loss_grad
    op = driver._op
    if type(driver).__name__ == "InfidelityOptimizer":
        kwargs = {"modulus":True, "normalize":True}
    else:
        kwargs = {}
    plot_projected_states(driver.state, op.target, U=op._U, e=e, g=g, name=type(driver).__name__, **kwargs)
    return True
    

def plot_projected_states(vs1, vs2, U=None, e=None, g=None, name=None, modulus=False, normalize=False, match_phase=False):
    if e is None:
        e = "e"
    else:
        e = np.real(e.mean)
        
    if g is None:
        g = "g"
    else:
        g = jnp.linalg.norm(jax.flatten_util.ravel_pytree(g)[0])
    
    if name is None:
        name = "n"
        
    v1 = vs1.to_array(normalize=normalize)
    v2 = make_Ustate(vs2, op=U, normalize=normalize)
    _plot_vec_and_close(v1, v2, modulus=modulus, match_phase=match_phase, name=name, e=e, g=g)
    
            
def _plot_vec_and_close(v1, v2, *, modulus, match_phase, name="", e="", g="", pause=0.1):
    plt.axhline(0, color='black')
    if modulus:
        plt.scatter(np.arange(v2.size), np.abs(v2), label="U@ref", alpha=0.7, color='green', lw=2, marker='o')
        plt.scatter(np.arange(v1.size), np.abs(v1), label="vstate", alpha=0.7, color='blue', lw=2, marker='x')
    else:
        if match_phase:
            v1 = remove_global_phase(v1)
            v2 = remove_global_phase(v2)
        plt.scatter(np.arange(v2.size), v2.real, label="U@ref", alpha=0.7, color='green', lw=2, marker='d')
        plt.scatter(np.arange(v2.size), v2.imag, alpha=0.7, color='green', lw=2, marker='o')
        plt.scatter(np.arange(v1.size), v1.real, label="vstate", alpha=0.7, color='blue', lw=2, marker='+')
        plt.scatter(np.arange(v1.size), v1.imag, alpha=0.7, color='blue', lw=2, marker='x')
    
    if not isinstance(e, str):
        e = "{:.14f}".format(e)
    if not isinstance(g, str):
        g = "{:.5f}".format(g)
    plt.title("{} : (e, g) = {} |  {}".format(name, e, g))
    
    plt.legend()
    plt.show(block=False) # needs latest mpl
    plt.pause(pause)
    plt.close()
    
def remove_global_phase(v):
    # compute the weighted phase
    v = v.astype(complex)
    prob = jnp.abs(v)**2
    prob /= prob.sum()
    idx = jnp.argmax(prob)
    ref_phase = jnp.angle(v[idx])
    return v*jnp.exp(-1j*ref_phase)

@jax.jit
def safe_log(x):
    return jnp.log(x+0j)



def driver_info(obj, depth=None):
    if hasattr(obj, "info") and callable(obj.info):
        return obj.info(depth)
    else:
        return str(obj)
    

def to_tuple(maybe_iterable):
    """
    to_tuple(maybe_iterable)

    Ensure the result is iterable. If the input is not iterable, it is wrapped into a tuple.
    """
    if hasattr(maybe_iterable, "__iter__"):
        surely_iterable = tuple(maybe_iterable)
    else:
        surely_iterable = (maybe_iterable,)

    return surely_iterable

def to_list(maybe_iterable):
    """
    to_list(maybe_iterable)

    Ensure the result is iterable. If the input is not iterable, it is wrapped into a list.
    """
    if maybe_iterable is None:
        surely_iterable = []
    elif hasattr(maybe_iterable, "__iter__"):
        surely_iterable = list(maybe_iterable)
    else:
        surely_iterable = [maybe_iterable]

    return surely_iterable


@jax.jit
def add_noise_to_param_dict(key, d, stddev=1e-5):
    leaves, tree_def = jax.tree_util.tree_flatten(d)
    n_leaves = len(leaves)
    keys = jax.random.split(key, n_leaves)
    key_tree = jax.tree_util.tree_unflatten(tree_def, keys)
    return jax.tree_util.tree_map(lambda x, k: x + stddev*jax.random.normal(k, shape=x.shape), d, key_tree)
    

def timer_to_dict(timer, _root=True):
    d = {}
    if _root:
        d["total_overview"] = str(timer)
    for k, v in timer.sub_timers.items():
        if len(v.sub_timers) == 0:
            v = v.total
        else:
            v = timer_to_dict(v, _root=False)
        d[k] = v
    return d

def timer_to_json(timer, path, indent=4):
    print_mpi("Outputting timer dict to:", path, flush=True)
    if mpi.node_number == 0:
        dir_name = os.path.dirname(path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)
        timer_dict = timer_to_dict(timer)
        with open(path, "w") as f:
            json.dump(timer_dict, f, indent=indent)
       
@jax.jit 
def safe_log(x):
    return jnp.log(x+0j)    

@partial(jax.jit, static_argnames=("axis", "keepdims"))
def mpi_logmeanexp_jax(a, axis=None, keepdims=False):
    """ Compute Log[Mean[Exp[a]]]"""
    # subtract logmax for better numerical stability
    a_max = mpi.mpi_max_jax(jnp.max(a.real, axis=axis, keepdims=True))[0]
    a_max = jax.lax.stop_gradient(a_max)
    a_shift = a - a_max
    exp_a = jnp.exp(a_shift)
    exp_a_mean = mpi.mpi_mean_jax(jnp.mean(exp_a, axis=axis, keepdims=True))[0]
    log_mean = safe_log(exp_a_mean) + a_max
    if not keepdims:
        log_mean = log_mean.squeeze(axis=axis)
    return log_mean

@partial(jax.jit, static_argnames=("axis", "keepdims"))
def mpi_logsumexp_jax(a, axis=None, keepdims=False):
    """ Compute Log[Mean[Exp[a]]]"""
    # subtract logmax for better numerical stability
    a_max = mpi.mpi_max_jax(jnp.max(a.real, axis=axis, keepdims=True))[0]
    a_max = jax.lax.stop_gradient(a_max)
    a_shift = a - a_max
    exp_a = jnp.exp(a_shift)
    exp_a_sum = mpi.mpi_sum_jax(jnp.sum(exp_a, axis=axis, keepdims=True))[0]
    log_sum = safe_log(exp_a_sum) + a_max
    if not keepdims:
        log_sum = log_sum.squeeze(axis=axis)
    return log_sum