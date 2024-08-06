from typing import Optional
import jax.numpy as jnp

from netket.operator import AbstractOperator, ContinuousOperator
from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.vqs import VariationalState, FullSumState, MCState
import warnings
from netket.stats import Stats
import numpy as np

from tre_tvmc.driver.infidelity.utils.sampling_Ustate import make_logpsi_U_afun
from tre_tvmc.driver.utils import copy_variational_state

class InfidelityOperatorUPsi(AbstractOperator):
    def __init__(
        self,
        state: VariationalState,
        *,
        U: AbstractOperator = None,
        cv_coeff: Optional[float] = None,
        U_dagger: AbstractOperator,
        dtype: Optional[DType] = None,
        sample_sqrt: bool = False,
    ):
        super().__init__(state.hilbert)

        if not isinstance(state, VariationalState):
            raise TypeError("The first argument should be a variational state.")

        if cv_coeff is not None:
            cv_coeff = jnp.array(cv_coeff)

            if (not is_scalar(cv_coeff)) or jnp.iscomplex(cv_coeff):
                raise TypeError("`cv_coeff` should be a real scalar number or None.")

            if isinstance(state, FullSumState):
                cv_coeff = None


        self._target = state
        self._cv_coeff = cv_coeff
        self._dtype = dtype

        self._U = U
        self._U_dagger = U_dagger
        self._wc = None
        self._sample_sqrt = sample_sqrt
                
    @property
    def target(self):
        return self._target

    @property
    def cv_coeff(self):
        return self._cv_coeff

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"InfidelityOperatorUPsi(target=U@{self.target}, U={self._U}, cv_coeff={self.cv_coeff}, importance_sampling={self._imp_sampling})"


def InfidelityUPsi(
    U: AbstractOperator,
    state: VariationalState,
    *,
    cv_coeff: Optional[float] = None,
    dtype: Optional[DType] = None,
    **kwargs,
):
    if not isinstance(U, ContinuousOperator):
        raise TypeError(
            "In order to sample from the state U|psi>, U must be"
            "an instance of ContinuousOperator."
        )
    # print("ABSORBING U INTO PHI")
    logpsiU, variables_U = make_logpsi_U_afun(state._apply_fun, U, state.variables)
    target = copy_variational_state(state, copy_samples=True, n_hot=1, variables=variables_U, apply_fun=logpsiU)

    return InfidelityOperatorUPsi(target, U=None, U_dagger=None, cv_coeff=cv_coeff, dtype=dtype, **kwargs)

def average_error_of_mean(R_list):
    means = jnp.array([R.mean for R in R_list])
    variances_of_mean = jnp.array([R.error_of_mean**2 for R in R_list])
    R_mean = jnp.mean(means)
    R_variance_of_mean = jnp.mean(variances_of_mean) / means.size
    R_error_of_mean = jnp.sqrt(R_variance_of_mean)
    return Stats(mean=R_mean, error_of_mean=R_error_of_mean)
