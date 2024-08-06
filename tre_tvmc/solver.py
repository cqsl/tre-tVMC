# import netket as nk
import jax.numpy as jnp
from functools import partial
import jax
from src.utils import real_dtype

# import copy
from netket.jax import tree_ravel
from netket.optimizer.qgt.qgt_jacobian_pytree import QGTJacobianPyTreeT
from netket.jax._utils_tree import RealImagTuple


def smooth_svd(Aobj, b, acond=1e-4, rcond=1e-2, exponent=6, x0=None):
    """
    From: https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.4.040302
    Solve the linear system using Singular Value Decomposition.
    The diagonal shift on the matrix should be 0.
    Internally uses {ref}`jax.numpy.linalg.lstsq`.
    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        rcond: The condition number
    """
    del x0

    A = Aobj.to_dense()

    b, unravel = tree_ravel(b)

    s2, V = jnp.linalg.eigh(A)
    del A  # memory saving

    b_tilde = V.T.conj() @ b

    svd_reg = _default_reg_fn(s2, rcond=rcond, acond=acond, exponent=exponent)

    cutoff = 10 * jnp.finfo(s2.dtype).eps
    s2_safe = jnp.maximum(s2, cutoff)
    reg_inv = svd_reg / s2_safe

    x = V @ (reg_inv * b_tilde)
    effective_rank = jnp.sum(svd_reg)

    info = {
        "effective_rank": effective_rank,
        "svd_reg": svd_reg,
        "s2": s2,
        "max_s2": jnp.max(s2),
    }

    del V  # memory saving

    return unravel(x), info


def _default_reg_fn(x, rcond, acond, exponent):
    cutoff = jnp.finfo(real_dtype(x.dtype)).eps

    if acond is not None:
        cutoff = jnp.maximum(cutoff, acond)

    cutoff = jnp.maximum(cutoff, rcond * jnp.max(x))

    return 1 / (1 + (cutoff / x) ** exponent)


def params_to_structure(parameters):
    pars_struct = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )
    return pars_struct


class block_solver:
    def __init__(self, solver, params, diag=False, **solver_kwargs):
        self.diag = diag
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        # if isinstance(params, Hashable):
        self.param_structure = params
        # else:
        #     self.param_structure = params_to_structure(params)
        # self.split_vmap = split_vmap # whether to take a vmapped block and separate it into separate blocks

    def __repr__(self):
        return f"block_solver(solver={self.solver}, diag={self.diag})"  # , split_vmap={self.split_vmap})"

    def __call__(self, A, b, x0=None):
        if not isinstance(A, QGTJacobianPyTreeT):
            raise NotImplementedError(
                f"block solver only implemented for QGTJacobianPyTree, but got: {A}"
            )

        # first make smaller qgt's per block
        # @jax.jit
        def _make_block_qgt(O_block, b_block, block_structure, x0_block, scale_block):
            """Contains a bit of hackary to avoid that netket tries to unravel.
            We make PyTrees of 1 element to do this.
            """
            block_qgt = QGTJacobianPyTreeT(
                O=[O_block],
                scale=[scale_block] if scale_block is not None else None,
                _params_structure=[block_structure],
                mode=A.mode,
                diag_shift=A.diag_shift,
            )
            if self.diag:  # hacky
                block_qgt = jnp.diag(1 / jnp.diag(block_qgt.to_dense()))
            block_x, block_info = self.solver(
                block_qgt,
                [b_block],
                x0=[x0_block] if x0_block is not None else None,
                **self.solver_kwargs,
            )
            block_x = block_x[0]  # undo the hack
            return block_x

        O = A.O
        scale = A.scale
        convert_back = False
        if isinstance(O, RealImagTuple):
            convert_back = True
        O, b, x0, scale = convert_reim_structure(O, b, x0, scale, self.param_structure)

        x = tree_map_realimag(
            _make_block_qgt,
            O,
            b,
            self.param_structure,
            x0,
            scale,
        )

        # map back to real imag tuple (at highest level)
        if convert_back:
            x = convert_reim_structure_back(x)
        return x, None


@jax.jit
def convert_reim_structure(O, b, x0, scale, param_structure):
    """Move the reim inwards to the nodes"""
    if isinstance(O, RealImagTuple):
        O = jax.tree_map(
            lambda *args: RealImagTuple(args), O.real, O.imag
        )  # combine per block
        b = jax.tree_map(
            lambda *args: RealImagTuple(args), b.real, b.imag
        )  # combine per block
        if x0 is not None:
            x0 = jax.tree_map(
                lambda *args: RealImagTuple(args), x0.real, x0.imag
            )  # combine per block
    if x0 is None:
        # not sure what this structure should really be?
        x0 = jax.tree_map(lambda _: None, param_structure)
    if scale is None:
        scale = jax.tree_map(lambda _: None, param_structure)
    return O, b, x0, scale


@jax.jit
def convert_reim_structure_back(x):
    return RealImagTuple(
        (tree_map_realimag(lambda a: a.real, x), tree_map_realimag(lambda a: a.imag, x))
    )


def tree_map_realimag(fn, *trees):
    return jax.tree_map(fn, *trees, is_leaf=lambda x: isinstance(x, RealImagTuple))


#


def split_tree(tree, axis=0):
    # we first see how large the axis is by looking at a leaf
    if isinstance(tree, jnp.ndarray):  # edge case
        return jnp.split(tree, tree.shape[axis], axis=axis)
    size = jax.tree_util.tree_leaves(tree)[0].shape[axis]
    trees = [
        jax.tree.map(lambda a: jnp.split(a, a.shape[axis], axis=axis)[i], tree)
        for i in range(size)
    ]
    return trees


@partial(jax.jit, static_argnames=("axis",))
def split_vmap_tree(tree, axis=0):
    if isinstance(tree, jnp.ndarray):
        return tree
    new_tree = {}
    for k, v in tree.items():
        if k.startswith("Vmap"):
            l = split_tree(v, axis=axis)
            l = {f"__{k}_{i}": vi for i, vi in enumerate(l)}
            new_tree[f"{k}"] = split_vmap_tree(l, axis=axis)
        else:
            new_tree[k] = split_vmap_tree(v, axis=axis)
    return new_tree


@partial(jax.jit, static_argnames=("axes",))
def split_vmap_in_trees(*trees, axes=0):
    if isinstance(axes, int):
        axes = (axes,) * len(trees)
    assert len(axes) == len(trees)
    trees = [split_vmap_tree(t, axis=a) for t, a in zip(trees, axes)]
    return trees


@partial(jax.jit, static_argnames=("axis",))
def unsplit_vmap_tree(tree, axis=0):
    if isinstance(tree, jnp.ndarray):
        return tree
    new_tree = {}
    if isinstance(tree, list):
        print(tree)
        return
    for k, v in tree.items():
        if k.startswith("Vmap"):
            ls = [v[f"__{k}_{i}"] for i in range(len(v))]  # make a list of trees
            l = jax.tree.map(lambda *tensors: jnp.concatenate(tensors, axis=axis), *ls)
            new_tree[f"{k}"] = unsplit_vmap_tree(l, axis=axis)
        else:
            new_tree[k] = unsplit_vmap_tree(v, axis=axis)
    return new_tree


@partial(jax.jit, static_argnames=("axes",))
def unsplit_vmap_in_trees(*trees, axes=0):
    if isinstance(axes, int):
        axes = (axes,) * len(trees)
    assert len(axes) == len(trees)
    trees = [unsplit_vmap_tree(t, axis=a) for t, a in zip(trees, axes)]
    return trees
