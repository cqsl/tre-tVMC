from typing import Callable, Optional, Union
from collections.abc import Sequence
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import flax
import os
import copy
import matplotlib.pyplot as plt
from functools import partial

import netket as nk
from netket import config
from netket.driver import AbstractVariationalDriver
from netket.driver.abstract_variational_driver import _to_iterable
from netket.jax import HashablePartial
from netket.logging.json_log import JsonLog
from netket.operator import AbstractOperator
from netket.utils import mpi
from netket.utils.dispatch import dispatch
from netket.utils.types import PyTree
from netket.vqs import VariationalState
from netket.graph import Graph
import netket.jax as nkjax
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.utils import mpi, timing

from tre_tvmc.operator import TREGenerator
from tre_tvmc.driver.utils import (
    copy_variational_state,
    print_mpi,
    driver_info,
    to_list,
    add_noise_to_param_dict,
)
from tre_tvmc.driver.callback import TrackBestModel
from .utils import default_inner_callback


class QDynamics(AbstractVariationalDriver):
    def __init__(
        self,
        operator: Union[Callable, AbstractOperator],
        variational_state: VariationalState,
        dt: float,
        *,
        t0: float = 0.0,
        optimizer: Callable = None,
        tevo_generator: TREGenerator = None,
        preconditioner: PreconditionerT = None,
        method: str = 'fidelity',
        method_kwargs: dict = None,
        n_samples_obs: int = None, # number of samples to use for evaluation of the observables, None means same as standard
        keep_best: bool = True,
        random_noise: float = None
    ):
        self.t0 = t0
        self.t = t0
        self.keep_best = keep_best
        self.random_noise = random_noise

        super().__init__(
            variational_state, optimizer=optimizer, minimized_quantity_name=""
        )
        self._generator_repr = repr(operator)
        if isinstance(operator, AbstractOperator):
            op = operator.collect()
            self.operator = lambda _: op
        else:
            self.operator = operator
            
        if tevo_generator is None:
            tevo_generator = TREGenerator(self.state.hilbert, operator, order=2)
        self.tevo_generator = tevo_generator
                
        self.method = method.lower()
        if method_kwargs is None:
            method_kwargs = {}
        self.method_kwargs = method_kwargs
        if preconditioner is None:
            preconditioner = identity_preconditioner
        self.preconditioner = preconditioner
        
        self.target_state = copy_variational_state(variational_state, n_hot=1, copy_samples=True, variables=variational_state.variables)
        
        if n_samples_obs is None or isinstance(variational_state, nk.vqs.FullSumState) or n_samples_obs == variational_state.n_samples:
            self._obs_state = None
        else:
            self._obs_state = copy_variational_state(variational_state, n_hot=0, variables=variational_state.variables, n_discard_per_chain=16) # increase discard per chain to burn in
            self._obs_state.n_samples = n_samples_obs
            # run hot now with the new samples
            for _ in range(2):
                self._obs_state.samples
                self._obs_state.reset()
            
        self.dt = dt

        self._loss_store = None
        self._propagator_and_dagger = None # will store the actual propagator for each projection

        self._stop_count = 0
        self._postfix = {}

    def _estimate_stats(self, observable):
        """
        We use a separate model for this part which might have more samples.
        """
        if self._obs_state is None:
            return self.state.expect(observable)
        else:
            self._obs_state.variables = self.state.variables
            self._obs_state.reset()
            return self._obs_state.expect(observable)
    
    def info(self, depth=0):
        lines = [
            f"{name}: {driver_info(obj, depth=depth + 1)}"
            for name, obj in [
                ("generator      ", self._generator_repr),
                ("state          ", self.state),
                ("dt             ", self.dt),
                ("method         ", self.method),
                ("preconditioner ", self.preconditioner),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self), *lines])

    def _project(self, n_iter, U=None, U_dagger=None, callback=None, **kwargs):
        with timing.timed_scope(name="dynamics_step_projection"):

            self.target_state.parameters = self.state.parameters
            if self.random_noise is not None:
                if hasattr(self.state, "samples"):
                    key = self.state.sampler_state.rng
                else:
                    key = jax.random.PRNGKey(0)
                self.target_state.parameters = add_noise_to_param_dict(key, self.target_state.parameters, stddev=self.random_noise)
            self.target_state.reset()

            callback = to_list(callback) # to avoid we change the list in place
            if len(callback) == 0:
                callback.append(default_inner_callback)
            if self.keep_best:
                tracker = TrackBestModel()
                callback.append(tracker)
                        
            if self.method in ("fidelity", "infidelity", "fid", "infid"):
                from .infidelity_psi import InfidelityOptimizer as InfidelityOptimizerImp
                proj_opt = InfidelityOptimizerImp(
                    self.target_state,
                    self.optimizer,
                    variational_state=self.state,
                    U=U, 
                    U_dagger=U_dagger,
                    preconditioner=self.preconditioner,
                    is_unitary=False,
                    sample_sqrt=False,
                    **self.method_kwargs,
                )
            else:
                raise Exception("projection method unknown.")
            proj_opt.run(n_iter, callback=callback, **kwargs)
            
            if self.keep_best:
                best_step, best_loss, best_params = tracker.get_best_step()
                if best_params is None:
                    if kwargs.get("show_progress", False):
                        print_mpi(f"Trying to take best params at step {best_step} with loss {best_loss}, but found None params", flush=True)
                    self.state.parameters = proj_opt.state.parameters
                else:                
                    if kwargs.get("show_progress", False):
                        print_mpi(f"Found best model at step {best_step} with loss {best_loss}", flush=True)
                    self.state.parameters = best_params
                self._loss_store.append(best_loss)
            else:            
                self._loss_store.append(proj_opt.loss)
                self.state.parameters = proj_opt.state.parameters

    def _advance_step(self, n_iter=250, show_inner_progress=False, out=None, inner_callback=None, minimal=False):
        if show_inner_progress:
            print_mpi("+"*100)
            print_mpi("t = ", self.t)
        self._loss_store = []
        
        for op_nr, (k, (op, op_dag, op_dt)) in enumerate(self.tevo_generator.taylor_root_expansion(self.t, dt=self.dt).items()):        
            op_type = k[0]
            self._propagator_and_dagger = (op, op_dag)
            self.state.reset()
            if show_inner_progress:
                print_mpi("Op = ", op)
            addendum = f"inner_t{self.t:.5f}_op{op_nr}"
            out_inner = os.path.join(out, addendum) if out is not None else addendum
            # create a json logger here to avoid that the parameters are being logged!!!
            out_inner = nk.logging.JsonLog(out_inner, save_params=False) # this will use less cpu
            self._project(n_iter, U=op, U_dagger=op_dag, out=out_inner, show_progress=show_inner_progress, callback=inner_callback)            
            
        self.t += self.dt
        print_mpi(f"Loss (t={self.t:.3f}):")
        print_mpi("-"*10)
        for i, loss in enumerate(self._loss_store):        
            print_mpi(f"L({i}) = ", loss.mean)
        print_mpi("-"*10, flush=True)
    
        
    def _iter(
        self,
        T: float,
        tstops: Optional[Sequence[float]] = None,
        callback: Optional[Callable] = None,
        **iter_kwargs
    ):
        """
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
        is called after every accepted step.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(f"All tstops must be in range [t, t + T]=[{self.t}, {T}]")

        if tstops is not None and len(tstops) > 0:
            tstops = np.sort(tstops)
            always_stop = False
        else:
            tstops = []
            always_stop = True

        while self.t < t_end:
            if always_stop or (
                len(tstops) > 0
                and (np.isclose(self.t, tstops[0]) or self.t > tstops[0])
            ):
                self._stop_count += 1
                yield self.t
                tstops = tstops[1:]

            # here comes the stepping
            with timing.timed_scope(name="dynamics_advance_step"):
                self._advance_step(**iter_kwargs)

            self._step_count += 1
            # optionally call callback
            with timing.timed_scope(name="dynamics_step_callback"):
                if callback:
                    callback()

        # Yield one last time if the remaining tstop is at t_end
        if (always_stop and np.isclose(self.t, t_end)) or (
            len(tstops) > 0 and np.isclose(tstops[0], t_end)
        ):
            yield self.t

    def run(
        self,
        T,
        out=None,
        out_inner=None,
        obs=None,
        n_iter=250,
        *,
        tstops=None,
        show_progress=True,
        show_inner_progress=True,
        callback=None,
        inner_callback=None,
        **iter_kwargs,
    ):
        """
        Runs the time evolution.

        By default uses :class:`netket.logging.JsonLog`. To know about the output format
        check it's documentation. The logger object is also returned at the end of this function
        so that you can inspect the results without reading the json output.

        Args:
            T: The integration time period.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing the observables that should be computed.
            tstops: A sequence of stopping times, each within the interval :code:`[self.t0, self.t0 + T]`,
                at which the driver will stop and perform estimation of observables, logging, and execute
                the callback function. By default, a stop is performed after each time step (at potentially
                varying step size if an adaptive integrator is used).
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to be executed at each
                stopping time.
        """
        if obs is None:
            obs = {}

        if callback is None:
            callback = lambda *_args, **_kwargs: True

        # Log only non-root nodes
        if self._mynode == 0:
            if out is None:
                loggers = ()
            # if out is a path, create an overwriting Json Log for output
            elif isinstance(out, str):
                loggers = (JsonLog(out, "w"),)
            else:
                loggers = _to_iterable(out)
        else:
            loggers = tuple()
            show_progress = False

        callbacks = _to_iterable(callback)
        callback_stop = False

        t_end = np.asarray(self.t + T)
        with tqdm(
            total=t_end,
            disable=not show_progress,
            unit_scale=True,
            dynamic_ncols=True,
            position=0
        ) as pbar:
            first_step = True

            # We need a closure to pass to self._iter in order to update the progress bar even if
            # there are no tstops
            def update_progress_bar():
                # Reset the timing of tqdm after the first step to ignore compilation time
                nonlocal first_step
                if first_step:
                    first_step = False
                    pbar.unpause()

                pbar.n = min(np.asarray(self.t), t_end)
                self._postfix["n"] = self.step_count
                self._postfix.update(
                    {
                        self._loss_name: str(self._loss_stats),
                    }
                )

                pbar.set_postfix(self._postfix)
                pbar.refresh()

            for step in self._iter(T, tstops=tstops, show_inner_progress=show_inner_progress, n_iter=n_iter, out=out_inner, callback=update_progress_bar, inner_callback=inner_callback, **iter_kwargs):
                with timing.timed_scope(name="dynamics_observables"):
                    log_data = self.estimate(obs)
                with timing.timed_scope(name="dynamics_additional_data"):
                    self._log_additional_data(log_data, step)

                self._postfix = {"n": self.step_count}
                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    self._postfix.update(
                        {
                            self._loss_name: str(self._loss_stats),
                        }
                    )
                    log_data[self._loss_name] = self._loss_stats
                pbar.set_postfix(self._postfix)


                # Execute callbacks before loggers because they can append to log_data
                with timing.timed_scope(name="dynamics_callbacks"):
                    for callback in callbacks:
                        if not callback(step, log_data, self):
                            callback_stop = True

                with timing.timed_scope(name="dynamics_loggers"):
                    for logger in loggers:
                        logger(self.step_value, log_data, self.state)

                if len(callbacks) > 0:
                    if mpi.mpi_any(callback_stop):
                        break
                update_progress_bar()


            # Final update so that it shows up filled.
            update_progress_bar()

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        return loggers

    def _log_additional_data(self, log_dict, step):
        log_dict["t"] = self.t
        if self._loss_store is not None and len(self._loss_store) > 0:
            for i, v in enumerate(self._loss_store):
                log_dict[f"inner_loss{i}"] = v
        log_dict["Generator"] = self.state.expect(self.operator(self.t))


    @property
    def step_value(self):
        return self.t

    def __repr__(self):
        return f"{type(self).__name__}(step_count={self.step_count}, t={self.t}, method={self.method})"
