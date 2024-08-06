from typing import Optional, Callable, Union
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils.types import DType, PyTree, Array
import netket.jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import HashableArray

class UFromHOperator(ContinuousOperator):
    r""" This is U = Exp[-1j H dt] \approx 1 - 1j*H*dt
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        H: ContinuousOperator,
        dt: float = 1e-2,
        dtype: Optional[DType] = jnp.complex128,
    ):
        r"""Args:
        hilbert: The underlying Hilbert space on which the operator is defined
        mass: float if all masses are the same, list indicating the mass of each particle otherwise
        """

        self.__attrs = None
        super().__init__(hilbert, dtype)

        self._dt = dt
        self._H = H.collect()
        self._is_hermitian = H.is_hermitian

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        dt, H_data = data
        Hloc = self._H._expect_kernel(logpsi, params, x, H_data)
        return 1 - 1j*dt*Hloc

    def _pack_arguments(self) -> PyTree:
        return (self._dt, self._H._pack_arguments())

    @property
    def _attrs(self):
        # used for the hashing
        if self.__attrs is None:
            self.__attrs = (self.hilbert, self.dtype, *self._H._attrs)
        return self.__attrs

    def __repr__(self):
        return f"UFromHOperator(H={self._H}, dt={self._dt})"