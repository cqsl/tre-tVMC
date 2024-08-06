from typing import Optional
from netket.stats import Stats

from netket.driver.abstract_variational_driver import AbstractVariationalDriver
from netket.callbacks import ConvergenceStopping
import numpy as np
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
import jax
import jax.numpy as jnp
import json
import flax
from netket.driver.abstract_variational_driver import apply_gradient
import copy
from typing import Any
from dataclasses import dataclass
import copy
import warnings

from tre_tvmc.driver.utils import (
    driver_info,
    to_tuple,
    print_mpi,
    copy_variational_state,
)
from tre_tvmc.driver.utils import copy_variational_state
from .logic import InfidelityOperator

@dataclass
class CheckPoint:
    """ This class will be used to store properties during the optimization.
    We will use it to backtrack when a given update was a bad one.
    """
    loss_stats: Stats
    variables: Any
    optimizer_state: Any  
    
    def __init__(self, loss_stats, vstate, optimizer_state):
        # make deepcopies to be certain
        # this should be VERY cheap to do, since it's the inner loop, so verify this
        # otherwise, we should store things ourselves:
        # - variables
        # - samples of the sampler
        self.loss_stats = loss_stats
        self.variables = copy.deepcopy(vstate.variables)
        self.optimizer_state = copy.deepcopy(optimizer_state)
        
    def reset_state(self, state, n_hot=1):
        # we might want to resample to avoid using the same bad samples from before
        state.variables = self.variables
        if hasattr(state, "samples"):
            for _ in range(n_hot):
                state.reset()
                state.samples
        return state

class InfidelityOptimizer(AbstractVariationalDriver):
    def __init__(
        self,
        target_state,
        optimizer,
        *,
        variational_state,
        U=None,
        U_dagger=None,
        preconditioner: PreconditionerT = identity_preconditioner,
        is_unitary=False,
        sample_Upsi=True,
        cv_coeff=-0.5,
        resample_target=True,
        n_sigma_check: float = None,
        n_redo: int = 1,
        sample_sqrt: bool = True,
    ):
        r"""
        Constructs a driver training the state to match the target state.

        The target state is either `math`:\ket{\psi}` or `math`:\hat{U}\ket{\psi}`
        depending on the provided inputs.

        Operator I_op computing the infidelity I among two variational states |ψ⟩ and |Φ⟩ as:

        .. math::

            I = 1 - |⟨ψ|Φ⟩|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩ = 1 - ⟨ψ|I_op|ψ⟩ / ⟨ψ|ψ⟩

        where:

         .. math::

            I_op = |Φ⟩⟨Φ| / ⟨Φ|Φ⟩

        The state |Φ⟩ can be an autonomous state |Φ⟩ =|ϕ⟩ or an operator U applied to it, namely
        |Φ⟩  = U|ϕ⟩. I_op is defined by the state |ϕ⟩ (called target) and, possibly, by the operator U.
        If U is not passed, it is assumed |Φ⟩ =|ϕ⟩.

        The Monte Carlo estimator of I is:

        ..math::

            I = \mathbb{E}_{χ}[ I_loc(σ,η) ] = \mathbb{E}_{χ}[ ⟨σ|Φ⟩ ⟨η|ψ⟩ / ⟨σ|ψ⟩ ⟨η|Φ⟩ ]

        where χ(σ, η) = |Ψ(σ)|^2 |Φ(η)|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩. In practice, since I is a real quantity, Re{I_loc(σ,η)}
        is used. This estimator can be utilized both when |Φ⟩ =|ϕ⟩ and when |Φ⟩ = U|ϕ⟩, with U a (unitary or
        non-unitary) operator. In the second case, we have to sample from U|ϕ⟩ and this is implemented in
        the function :ref:`jax.:ref:`InfidelityUPsi`. This works only with the operators provdided in the package.
        We remark that sampling from U|ϕ⟩ requires to compute connected elements of U and so is more expensive
        than sampling from an autonomous state. The choice of this estimator is specified by passing
        `sample_Upsi=True`, while the flag argument `is_unitary` indicates whether U is unitary or not.

        If U is unitary, the following alternative estimator can be used:

        ..math::

            I = \mathbb{E}_{χ'}[ I_loc(σ,η) ] = \mathbb{E}_{χ}[ ⟨σ|U|ϕ⟩ ⟨η|ψ⟩ / ⟨σ|U^{\dagger}|ψ⟩ ⟨η|ϕ⟩ ].

        where χ'(σ, η) = |Ψ(σ)|^2 |ϕ(η)|^2 / ⟨ψ|ψ⟩ ⟨ϕ|ϕ⟩. This estimator is more efficient since it does not
        require to sample from U|ϕ⟩, but only from |ϕ⟩. This choice of the estimator is the default and it works only
        with `is_unitary==True` (besides `sample_Upsi=False`). When |Φ⟩ = |ϕ⟩ the two estimators coincides.

        To reduce the variance of the estimator, the Control Variates (CV) method can be applied. This consists
        in modifying the estimator into:

        ..math::

            I_loc^{CV} = Re{I_loc(σ,η)} - c (|1 - I_loc(σ,η)^2| - 1)

        where c ∈ \mathbb{R}. The constant c is chosen to minimize the variance of I_loc^{CV} as:

        ..math::

            c* = Cov_{χ}[ |1-I_loc|^2, Re{1-I_loc}] / Var_{χ}[ |1-I_loc|^2 ],

        where Cov[..., ...] indicates the covariance and Var[...] the variance. In the relevant limit
        |Ψ⟩ →|Φ⟩, we have c*→-1/2. The value -1/2 is adopted as default value for c in the infidelity
        estimator. To not apply CV, set c=0.

        Args:
            target_state: target variational state |ϕ⟩.
            optimizer: the optimizer to use to use (from optax)
            variational_state: the variational state to train
            U: operator U.
            U_dagger: dagger operator U^{\dagger}.
            cv_coeff: Control Variates coefficient c.
            is_unitary: flag specifiying the unitarity of U. If True with `sample_Upsi=False`, the second estimator is used.
            dtype: The dtype of the output of expectation value and gradient.
            sample_Upsi: flag specifiying whether to sample from |ϕ⟩ or from U|ϕ⟩. If False with `is_unitary=False`, an error occurs.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        
        # do some changes first
        if sample_sqrt:
            variational_state = make_psi_state(variational_state)
            target_state = make_psi_state(target_state)
        
        
        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Infidelity"
        )

        self._cv = cv_coeff
        self.resample_Upsi = resample_target
        self.n_sigma_check = n_sigma_check
        self.n_redo = n_redo
        # will contain the info of last good step to check in next step
        # should ideally also store the samples (!)
        self._checkpoint = None 
        self._preconditioner = preconditioner
        self._sample_sqrt = sample_sqrt

        self._op = InfidelityOperator(
            target_state,
            U=U,
            U_dagger=U_dagger,
            is_unitary=is_unitary,
            cv_coeff=cv_coeff,
            sample_Upsi=sample_Upsi,
            sample_sqrt = sample_sqrt,
        )
        
        # we HAVE to let them adjust a bit before we start
        for _ in range(8):
            self._op.target.reset()
            self._op.target.samples
            self.state.reset()
            self.state.samples
            

    def _forward_and_backward(self):
        """ this now also performs a check to see if we update or not (!). """     
        self.state.reset()
        if self.resample_Upsi:
            self._op.target.reset()
            
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self._op)
        
        self._dp = self.preconditioner(self.state, self._loss_grad, self.step_count)
        
        return self._dp
    
    def iter(self, n_steps: int, step: int = 1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            n_steps: The total number of steps to perform (this is
                equivalent to the length of the iterator)
            step: The number of internal steps the simulation
                is advanced between yielding from the iterator

        Yields:
            int: The current step.
        """
        n_seq_failed = 0
        max_seq_failed = self.n_redo
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                dp = self._forward_and_backward()
                new_stats = self._loss_stats
                
                if self._checkpoint is None:
                    # initialize first time
                    self._checkpoint = CheckPoint(self._loss_stats, self.state, self._optimizer_state)
                
                # perform some checks to see whether we need to backtrack (!!!)
                # cast into some useful variable names
                prev_stats = self._checkpoint.loss_stats
                backtrack = False
                # add additional flexibility due to bad error estimate now
                # this only works for INFIDELITY CENTERED AROUND 0 (!!!)
                # if self._step_count > 0 and \
                if n_seq_failed <= max_seq_failed and \
                        new_stats.mean > prev_stats.mean + self.n_sigma_check*np.abs(prev_stats.mean) + self.n_sigma_check*prev_stats.error_of_mean:
                    print_mpi(f"Update did not meet the requirements (step={self.step_count}):", new_stats, "while previous = ", prev_stats, flush=True)
                    backtrack = True
                    n_seq_failed += 1
                else:
                    n_seq_failed = 0
                    
                if i == 0: # only at first step they compute things
                    yield self.step_count
                self._step_count += 1
                
                if backtrack:
                    print_mpi("BACKTRACK!", flush=True)
                    self._backtrack_parameters()
                else:
                    # this value becomes the new reference
                    self._checkpoint = CheckPoint(self._loss_stats, self.state, self._optimizer_state)
                    self.update_parameters(dp)

    def _backtrack_parameters(self):
        # reset the state to what made sense before
        print_mpi("Backtracking the parameters", flush=True)
        self._checkpoint.reset_state(self.state, n_hot=1)
        self._optimizer_state = self._checkpoint.optimizer_state       
        

    def run(
        self,
        n_iter,
        out=None,
        *args,
        target_infidelity=None,
        callback=lambda *x: True,
        **kwargs,
    ):
        """
        Executes the Infidelity optimisation, updating the weights of the network
        stored in this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`. If no logger is specified, creates a json file at `out`,
        overwriting files with the same prefix.

        Args:
            n_iter: the total number of iterations
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
            obs: An iterable containing all observables that should be computed
            target_infidelity: An optional floating point number that specifies when to stop the optimisation.
                This is used to construct a {class}`netket.callbacks.ConvergenceStopping` callback that stops
                the optimisation when that value is reached. You can also build that object manually for more
                control on the stopping criteria.
            step_size: Every how many steps should observables be logged to disk (default=1)
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to stop training given a condition
        """
        callbacks = to_tuple(callback)

        if target_infidelity is not None:
            cb = ConvergenceStopping(
                target_infidelity, smoothing_window=20, patience=30
            )
            callbacks = callbacks + (cb,)

        super().run(n_iter, out, *args, callback=callbacks, **kwargs)

    @property
    def cv(self) -> Optional[float]:
        """
        Return the coefficient for the Control Variates
        """
        return self._cv

    @property
    def infidelity(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats
    
    @property
    def loss(self) -> Stats:
        return self.infidelity

    @property
    def preconditioner(self):
        """
        The preconditioner used to modify the gradient.

        This is a function with the following signature

        .. code-block:: python

            precondtioner(vstate: VariationalState,
                          grad: PyTree,
                          step: Optional[Scalar] = None)

        Where the first argument is a variational state, the second argument
        is the PyTree of the gradient to precondition and the last optional
        argument is the step, used to change some parameters along the
        optimisation.

        Often, this is taken to be :func:`nk.optimizer.SR`. If it is set to
        `None`, then the identity is used.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, val: Optional[PreconditionerT]):
        if val is None:
            val = identity_preconditioner

        self._preconditioner = val

    def __repr__(self):
        return (
            "InfidelityOptimiser("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state},"
            + f"\n  sample_sqrt = {self._sample_sqrt})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, driver_info(obj, depth=depth + 1))
            for name, obj in [
                ("Propagator    ", self._op.U),
                ("Optimizer      ", self._optimizer),
                ("Preconditioner ", self.preconditioner),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)


def dryrun_update_parameters(p, dp, optimizer, optimizer_state):
    """
    Updates the parameters of the machine using the optimizer in this driver

    Args:
        dp: the pytree containing the updates to the parameters
    """
    new_state, new_params = apply_gradient(
        optimizer.update, optimizer_state, dp, p
    )
    return new_state, new_params


def make_psi_state(vs, n_hot=5):
    new_vs = copy_variational_state(vs, n_hot=0, copy_samples=True)
    # following is a bit hacky, but otherwise the samples are completely reinitialized.
    new_vs._sampler = new_vs.sampler.replace(
        machine_pow=1,    
    ) # simpler change
    for _ in range(n_hot):
        new_vs.reset()
        new_vs.samples
    new_vs.reset()
    return new_vs
