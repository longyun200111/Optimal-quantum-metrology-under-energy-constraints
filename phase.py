'''
Calculate the optimal average Holevo cost for the phase estimation problem of different dimensions
with arbitrary (fixed) classical estimators.
'''

from cvxpy import constraints, partial_trace
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import itertools
from constants import get_H

from joblib import Parallel, delayed, Memory
from tqdm import tqdm

from utils import *

memory = Memory(location='./cachedir', verbose=0)

@memory.cache
def get_J(dim):
    J = np.zeros((dim**2, dim**2), dtype=complex)
    ket = [
            # |i>
            sp.coo_matrix(([1.], ([i], [0])), shape=(dim, 1)).toarray()
            for i in range(dim)
        ]
    for i in range(dim-1):
        J += np.kron(ket[i] @ ket[i+1].T, ket[i] @ ket[i+1].T)
    return J

@memory.cache
def construct_prob(dim):
    Tx = [
        cp.Variable((dim**2, dim**2), hermitian=True)
        for _ in range(dim)
    ]
    rho = cp.Variable((dim, dim), hermitian=True)

    lambda_x = [cp.Parameter(complex=True) for _ in range(dim)]
    E = cp.Parameter() 

    J = get_J(dim)

    obj = cp.Minimize(2 - sum(
        cp.real(2 * lambda_x[i] * cp.trace(Tx[i] @ J))
        for i in range(dim)
    ))

    T = sum(
        cp.kron(Tx[i], fock_dm(dim, i))
        for i in range(dim)
    )

    constraints = [ T >> 0 ]
    constraints += [ sum(Tx) == cp.kron(rho, np.eye(dim))]
    constraints += [ cp.trace(rho) == 1, rho >> 0 ]

    H = get_H(dim)
    expanded_H = np.kron(H, np.eye(dim*dim)) + np.kron(np.eye(dim*dim), H)
    constraints += [
        cp.partial_trace(
            cp.partial_trace(expanded_H @ T, [dim]*3, 0),
            [dim]*2, 1
        ) << H.T + E * np.eye(dim)
    ]
    constraints += [ cp.real(cp.trace(H @ rho)) <= E ]

    prob = cp.Problem(obj, constraints)

    try:
        prob.get_problem_data(solver=cp.MOSEK)
    except Exception:
        pass

    return prob, lambda_x, E

def calc(args, dim=2):
    lambda_x_val = args[:-1]
    E_val = args[-1]

    prob, lambda_x, E = construct_prob(dim)
    for i in range(dim):
        lambda_x[i].value = lambda_x_val[i]
    E.value = E_val

    prob.solve(solver=cp.MOSEK)

    sol = dict()
    sol['val'] = prob.value
    sol['energy'] = E_val
    
    return sol

def solve(E, dim=2, is_fixed=False):
    if is_fixed:
        lambda_vals = [np.exp(2j*i*np.pi/dim) for i in range(dim)]
        pairs = list(itertools.product(*[lambda_vals for _ in range(dim)], E))
    else:
        lambda_vals = np.exp(1j * np.linspace(0, 2 * np.pi, 20))
        pairs = list(itertools.product(*[lambda_vals for _ in range(dim)], E))
    from functools import partial
    res = Parallel(n_jobs=-1)(delayed(partial(calc, dim=dim))(pair) for pair in tqdm(pairs))

    opt_val = [
        min(
            sol['val']
            for sol in res
            if sol['energy'] == e
        )
        for e in E    
    ]

    return opt_val

if __name__ == '__main__':
    E = np.linspace(0, 2, 20)

    cost2 = solve(E, 2, is_fixed=False)
    cost3 = solve(E, 3, is_fixed=False)
    cost3_fix = solve(E, 3, is_fixed=True)

    plt.plot(E, cost2, label='d=2')
    plt.plot(E, cost3_fix, label='d=3 fixed estimator')
    plt.plot(E, cost3, label='d=3 arbitrary estimator')
    plt.xlabel('E')
    plt.ylabel('average cost')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('phase.pdf', dpi = 600)
    plt.show()