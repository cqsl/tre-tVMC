import numpy as np
import jax.numpy as jnp
import jax
from netket.stats import Stats

from tre_tvmc.driver.utils import print_mpi


class TrackBestModel:
    def __init__(self, monitor: str = "mean"):
        self.monitor = monitor
        self.best_loss_value = np.inf
        self.best_loss_stats = None
        self.best_params = None
        self.best_step = 0

    def __call__(self, step, log_data, driver):
        loss_value = np.real(getattr(log_data[driver._loss_name], self.monitor))
        # below start step we always keep the last model, after only if it improves
        if loss_value < self.best_loss_value:
            self.best_loss_value = loss_value
            self.best_loss_stats = log_data[driver._loss_name]
            self.best_params = driver.state.parameters
            self.best_step = step
        return True

    def get_best_step(self):
        return self.best_step, self.best_loss_stats, self.best_params


def acceptance_callback(step_nr, log_data, driver):
    if not hasattr(driver.state, "sampler_state"):  # FullSumState
        return True
    if (
        hasattr(driver.state.sampler_state, "acceptance")
        and driver.state.sampler_state.acceptance is not None
    ):
        log_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True


class EarlyStopping:
    """A simple callback to stop NetKet if there are no more improvements in the training.
    based on `driver._loss_name`.
    """

    def __init__(self, target: float, monitor: str = "mean", min_steps: int = 10):
        self.target = target
        self.monitor = monitor
        self.min_steps = min_steps

    def __call__(self, step, log_data, driver):
        """
        A boolean function that determines whether or not to stop training.

        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            driver: A NetKet variational driver.

        Returns:
            A boolean. If True, training continues, else, it does not.
        """

        loss = np.real(getattr(log_data[driver._loss_name], self.monitor))

        if step > self.min_steps and loss < self.target:  # stop
            print_mpi(
                f"reached target loss (after n_steps={step}  >{self.min_steps}): {loss} (< target={self.target})",
                flush=True,
            )
            return False
        else:
            return True


def nans_callback(step_nr, log_data, driver):
    state = driver.state

    def _not_nan(x):
        return np.all(~np.isnan(x))

    tree_not_nan = jax.tree_util.tree_map(_not_nan, state.parameters)
    cont = jax.tree_util.tree_all(tree_not_nan)
    if not cont:
        print_mpi("Found nan parameters:")
        print_mpi(state.parameters)

    if cont and hasattr(driver, "_loss_stats") and driver._loss_stats is not None:
        cont = ~np.isnan(driver._loss_stats.mean)
        if not cont:
            print_mpi("Found nan loss stats", driver._loss_stats)

    return cont


class EarlyStoppingEMA:
    """An exponential-moving-average-based callback to stop NetKet if there are no more improvements in the training.
    Based on `driver._loss_name`."""

    def __init__(
        self,
        min_delta=0.0,
        patience=25,
        monitor="mean",
        skip_first_n=50,
        alpha=0.95,
        logspace=False,
    ):
        self.min_delta = min_delta
        self.patience = patience
        self.monitor = monitor
        self.alpha = alpha
        self.skip_first_n = skip_first_n
        self.logspace = logspace
        self._best_val = np.infty
        self._best_iter = 0

    def _reset(self):
        self._best_val = np.infty
        self._best_iter = 0
        self._val = None

    def _update(self, step, loss):
        if isinstance(loss, (jnp.ndarray, np.ndarray)):
            loss = loss.mean()
        elif isinstance(loss, Stats):
            loss = loss.mean
        nval = np.real(loss)

        if self.logspace:
            nval = np.log10(nval)

        if self._val is None:
            self._val = nval  # better first value
        self._val = self.alpha * nval + (1 - self.alpha) * self._val

        if self._val <= self._best_val:
            self._best_val = self._val
            self._best_iter = step

    def __call__(self, step, log_data, driver):
        """
        A boolean function that determines whether or not to stop training.
        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            driver: A NetKet variational driver.
        Returns:
            A boolean. If True, training continues, else, it does not.
        """
        loss = log_data[driver._loss_name]
        if step < self.skip_first_n:  # do not even set _val
            self._reset()  # this will help when doing multiple optimizations, without the need to call reset explicitly when restarting!
            return True  # skip

        self._update(step, loss)

        if (
            step - self._best_iter >= self.patience
            and self._val > self._best_val - self.min_delta
        ):
            print_mpi(
                f"stop because converged (after n_steps={step}): {loss}", flush=True
            )
            return False
        else:
            return True
