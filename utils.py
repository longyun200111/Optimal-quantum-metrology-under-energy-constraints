import scipy.sparse as sp
import numpy as np
import cvxpy as cp

def ptrace(rho, dims, axes):
    """
    Partial trace over the specified axis of a density matrix.
    Work for numpy & cvxpy. 
    """
    dims = np.reshape(dims, (-1,)).tolist()
    traced_rho = rho

    if isinstance(axes, int):
        axes = [axes]

    for axis in axes:
        if axis < 0:
            axis += len(dims)
        left_id = sp.eye(np.prod(dims[:axis]))
        right_id = sp.eye(np.prod(dims[axis+1:]))
        basis = [
            # |i>
            sp.coo_matrix(([1.], ([i], [0])), shape=(dims[axis], 1))
            for i in range(dims[axis])
        ]
        
        traced_rho = sum(
            # (I ⊗ <i| ⊗ I) . (I ⊗ |i> ⊗ I)
            sp.kron(left_id, sp.kron(v.T, right_id)) @ traced_rho @ sp.kron(left_id, sp.kron(v, right_id))
            for v in basis
        )

        dims[axis] = 1
    
    return traced_rho


def ptrans(rho, dims, axes):
    """
    Partial transpose over the specified axis of a density matrix.
    Work for numpy & cvxpy. 
    """
    dims = np.reshape(dims, (-1,)).tolist()
    trans_rho = rho

    if isinstance(axes, int):
        axes = [axes]

    for axis in axes:
        if axis < 0:
            axis += len(dims)
        left_id = sp.eye(np.prod(dims[:axis]))
        right_id = sp.eye(np.prod(dims[axis+1:]))
        basis = [
            # |i><j|
            sp.coo_matrix(([1.], ([i], [j])), shape=(dims[axis], dims[axis]))
            for i in range(dims[axis])
            for j in range(dims[axis])
        ]
        
        trans_rho = sum(
            # (I ⊗ |i><j| ⊗ I) . (I ⊗ |i><j| ⊗ I)
            sp.kron(left_id, sp.kron(v, right_id)) @ trans_rho @ sp.kron(left_id, sp.kron(v, right_id))
            for v in basis
        )
    
    return trans_rho


def permute(rho, dims, perm):
    """
    Permute the axes of a density matrix.
    Work for numpy & cvxpy. 
    """
    dims = np.reshape(dims, (-1,)).tolist()
    perm_dims = [dims[perm[i]] for i in range(len(perm))]

    d = np.prod(dims)
    row = []
    col = []
    for i in range(d):
        coord = np.unravel_index(i, dims)
        permuted_coord = np.ravel_multi_index([coord[perm[j]] for j in range(len(perm))], perm_dims)
        # |i><perm_i|
        row.append(i)
        col.append(permuted_coord)
    perm_mul = sp.coo_matrix(([1.]*d, (row, col)), shape=(d, d))

    perm_rho = perm_mul.T @ rho @ perm_mul
    
    return perm_rho


def extend(op, dims, axes):
    """
    Extend an operator to a larger system
    """
    dims = np.reshape(dims, (-1,)).tolist()
    if isinstance(axes, int):
        axes = [axes]
    remaining_dims = [dims[i] for i in range(len(dims)) if i not in axes]

    if isinstance(op, cp.Expression):
        ext_op = cp.kron(op, sp.eye(np.prod(remaining_dims)))
    else:
        ext_op = sp.kron(op, sp.eye(np.prod(remaining_dims)))

    ext_dims = [dims[i] for i in range(len(dims)) if i in axes] + remaining_dims

    perm = [None for _ in range(len(dims))]
    for i in range(len(axes)):
        perm[axes[i]] = i
    j = 0
    for i in range(len(axes), len(dims)):
        while perm[j] is not None:
            j += 1
        perm[j] = i

    return permute(ext_op, ext_dims, perm)


def partial_comb(comb, dims, n):
    """
    Partial quantum comb for first n steps (n=1,2,...,N)

    dims: [(dim1, dim2), (dim3, dim4), ...]
    each tuple represents a step
    """
    N = len(dims)
    traced_comb = ptrace(comb, dims, list(range(2*n,2*N)))
    traced_comb = traced_comb / np.prod([dims[i][0] for i in range(n, N)])
    return traced_comb

def fock_dm(dim, i):
    """
    Generate a Fock state density matrix.
    """
    dm = np.zeros((dim, dim), dtype=complex)
    dm[i, i] = 1
    return dm
