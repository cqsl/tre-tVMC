import numpy as np


from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator, AbstractOperator

from .operator.tevo import UFromHOperator

COEFFS = [
    (),
    (1,),
    ((1 + 1j) / 2, (1 - 1j) / 2),
    (
        0.18673085336460016 + 0.4807738845503311j,
        0.18673085336460016 - 0.4807738845503311j,
        0.6265382932707997 + 0j,
    ),
    (
        0.042626656502702476 + 0.39463295317211333j,
        0.042626656502702476 - 0.39463295317211333j,
        0.4573733434972975 + 0.23510048799854277j,
        0.4573733434972975 - 0.23510048799854277j,
    ),
]


class TREGenerator:
    """The core of the TRE method: yields sequences of Taylor-Root Expansion operators."""

    def __init__(self, hilbert: AbstractHilbert, H: ContinuousOperator, order: int = 2):
        self.hilbert = hilbert
        if isinstance(H, AbstractOperator):
            op = H.collect()
            self.H = lambda _: op
        else:
            self.H = H
        assert order >= 0
        assert order <= len(COEFFS)
        self.coeffs = COEFFS[order]
        self.order = order

        # internal check
        c = np.sum(self.coeffs)
        assert np.isclose(
            c, 1
        ), f"got order that gave wrong solution: {self.order} with sum = {c} != 1"

    def taylor_root_expansion(self, t, dt=1e-2):
        op_dict = {}
        Ht = self.H(t)
        for i, ck in enumerate(self.coeffs):
            # (k, (op, op_dag, op_dt))
            k = f"Uk({i})"
            op = UFromHOperator(self.hilbert, Ht, dt=ck * dt)
            op_dag = UFromHOperator(self.hilbert, Ht, dt=-ck * dt)
            op_dt = dt
            op_dict[k] = (op, op_dag, op_dt)
        return op_dict

    def __repr__(self):
        return f"TREGenerator(K={self.order})"
