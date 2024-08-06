# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# this example can be opened as a notebook through jupytext
# -

import netket as nk
import flax.linen as nn
import jax
import jax.numpy as jnp
from netket.utils.types import DType

# +
# implements a Harmonic Oscillator with a single parameters

# +
hi = nk.hilbert.Particle(1, D=1, pbc=False)

ekin = nk.operator.KineticEnergy(hi, mass=1)


def vfun(x):
    return 0.5 * jnp.sum(x**2)


epot = nk.operator.PotentialEnergy(hi, vfun)

ham = ekin + epot
ham_t = lambda t: ham


# +
class HOModel(nn.Module):
    param_dtype: DType = jnp.complex128

    # See thesis of Giuseppe Carleo
    # https://iris.sissa.it/retrieve/dd8a4bf7-04b2-20a0-e053-d805fe0a8cb0/1963_5357_carleo.pdf#page=94.12
    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1)  # single particle in 1d
        a = self.param("alpha", jax.nn.initializers.ones, (), self.param_dtype)
        return -a * x**2 / 2


ma = HOModel()
ma
# -

sa = nk.sampler.MetropolisGaussian(hi, sigma=1.0, sweep_size=hi.size * 2, n_chains=128)
vs = nk.vqs.MCState(sa, ma, n_samples=8 * 1024, n_discard_per_chain=4)
vs

vs.samples.shape

vs.sampler_state.acceptance

from tre_tvmc.driver import QDynamics
from tre_tvmc.tre import TREGenerator
from tre_tvmc.solver import block_solver
from jax.scipy.sparse.linalg import cg

# +
tre_generator = TREGenerator(hi, ham, order=4)
dt = 1e-2
Tmax = 1

solver = block_solver(cg, vs.parameters)
qgt = nk.optimizer.qgt.QGTJacobianPyTree
sr = nk.optimizer.SR(qgt, holomorphic=True, diag_shift=1e-3)
opt = nk.optimizer.Sgd(1e-2)

qgt
# -

method_kwargs = {
    "sample_Upsi": False,
    "n_sigma_check": 0.1,
    "n_redo": 20,
    "cv_coeff": -0.5,
}
te = QDynamics(
    ham_t,
    vs,
    dt,
    tevo_generator=tre_generator,
    preconditioner=sr,
    optimizer=opt,
    method_kwargs=method_kwargs,
)

te.run(
    Tmax, n_iter=25, out="runs/test", out_inner="runs/test/", show_inner_progress=False
)
