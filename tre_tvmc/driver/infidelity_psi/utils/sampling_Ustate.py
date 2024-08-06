import jax
import flax
from netket import jax as nkjax

from tre_tvmc.driver.utils import safe_log

def make_logpsi_U_afun(logpsi_fun, U, variables):
    """Wraps an apply_fun into another one that multiplies it by an
    Unitary transformation U.

    This wrapper is made such that the Unitary is passed as the model_state
    of the new wrapped function, and therefore changes to the angles/coefficients
    of the Unitary should not trigger recompilation.

    Args:
        logpsi_fun: a function that takes as input variables and samples
        U: a {class}`nk.operator.JaxDiscreteOperator`
        variables: The variables used to call *logpsi_fun*

    Returns:
        A tuple, where the first element is a new function with the same signature as
        the original **logpsi_fun** and a set of new variables to be used to call it.
    """
    # wrap apply_fun into logpsi logpsi_U
    logpsiU_fun = nkjax.HashablePartial(_logpsi_U_fun, logpsi_fun)

    # Insert a new 'model_state' key to store the Unitary. This only works
    # if U is a pytree that can be flattened/unflattened.
    new_variables = flax.core.copy(variables, {"unitary": U})

    return logpsiU_fun, new_variables


def _logpsi_U_fun(apply_fun, variables, x, *args):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `unitary` with
    a jax-compatible operator.
    """
    variables_applyfun, U = flax.core.pop(variables, "unitary")

    logpsi_x = apply_fun(variables_applyfun, x, *args)
    logUlocal_x = safe_log(U._expect_kernel(apply_fun, variables_applyfun, x, U._pack_arguments()))
    logUpsi_x = logUlocal_x + logpsi_x
    return logUpsi_x    
