from functools import partial

import jax
import jax.numpy as jnp
# from jax.scipy.special import logsumexp
from netket.jax import logsumexp_cplx as logsumexp

from netket import jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.vqs import MCState, expect, expect_and_forces, get_local_kernel_arguments, get_local_kernel
from netket.utils import mpi
from typing import Callable, Optional
from netket.utils.types import PyTree, Array
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from .operator import InfidelityOperatorUPsi
from tre_tvmc.driver.utils import calc_full_chunk_size

@expect.dispatch    
def infidelity(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: None):
    # temporarily set and unset
    chunk_size = calc_full_chunk_size(vstate)
    vstate.chunk_size = chunk_size
    out = expect(vstate, op, chunk_size)
    vstate.chunk_size = None
    return out

@expect_and_forces.dispatch
def infidelity_and_forces(
    vstate: MCState,
    op: InfidelityOperatorUPsi,
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
    ):
    # temporarily set and unset
    chunk_size = calc_full_chunk_size(vstate)
    vstate.chunk_size = chunk_size
    out = expect_and_forces(vstate, op, chunk_size, mutable=mutable)
    vstate.chunk_size = None
    return out