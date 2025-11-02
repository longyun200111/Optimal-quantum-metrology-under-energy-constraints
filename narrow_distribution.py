'''
Calculate the average Holevo cost with a narrow prior distribution for the phase channel
'''

import cvxpy as cp
import scipy.linalg
import itertools
import matplotlib.pyplot as plt
from constants import get_H
import numpy as np
import scipy.sparse as sp
from scipy import integrate
import sympy
from functools import cache
import csv
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed, Memory

from utils import *
from numeric_utils import *

memory = Memory('cachedir', verbose=0)

def pdf(x, mu, sigma):
    # mu = 1
    # sigma = 0.1
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

@memory.cache
def construct_J(mu, sigma):
    theta = sympy.symbols('theta')

    Omega = sympy.Matrix([
        [1],
        [0],
        [0],
        [1]
    ])

    U_theta = sympy.Matrix([
        [1, 0],
        [0, sympy.exp(sympy.I * theta)]
    ])

    C_theta = sympy.kronecker_product(sympy.eye(2), U_theta) @ Omega @ Omega.H @ sympy.kronecker_product(sympy.eye(2), U_theta).H
    C_theta = sympy.lambdify(theta, C_theta, 'numpy')

    integrated = lambda x: np.exp(-1j * x) * C_theta(x).T * pdf(x, mu, sigma)

    J = integrate(
        integrated,
        -np.inf, np.inf,
        min_k = 13,
        max_k = 20
    )

    # J = sympy.integrate(integrated, (theta, 0, 2 * np.pi))

    return np.array(J, dtype=complex)

@memory.cache
def construct_prob(dim = 2, mu=np.pi, sigma=0.1):
    avg_cost = 0
    conds = []

    lambda_x_real = [cp.Parameter() for _ in range(dim)]
    lambda_x_imag = [cp.Parameter() for _ in range(dim)]
    E = cp.Parameter() 

    comb_dim = [(1, dim), (dim, dim)]

    mul_val = [None for _ in range(dim)]

    N = len(comb_dim)

    T_x = [cp.Variable((np.prod(comb_dim) // dim, np.prod(comb_dim) // dim), hermitian=True) for _ in range(dim)]
    T = sum(
        cp.kron(T_x[i], sp.coo_matrix(([1], ([i], [i])), shape=(dim, dim)) )
        for i in range(dim)
    )
    partial_T = [np.ones((1, 1))] + [
            partial_comb(T, comb_dim, n)
            for n in range(1, N+1)
        ]

    # comb constraints
    conds += [
        ptrace(partial_T[n], comb_dim[:n], -1) == cp.kron(partial_T[n-1], np.eye(comb_dim[n-1][0]))
        for n in range(1, N+1)
    ]
    conds += [
        T >> 0
    ]
        
    # objective
    J = construct_J(mu, sigma)
    for i in range(dim):
        J_comp = np.kron(J, 
            fock_dm(2, i)
        )
        mul_val[i] = cp.trace(T @ J_comp)
        avg_cost += 2 * lambda_x_real[i] * cp.real(mul_val[i]) - 2 * lambda_x_imag[i] * cp.imag(mul_val[i])

    # energy constraints
    T_sup_energy = T
    H = lambda d : get_H(d)
    dim_in = [comb_dim[i][0] for i in range(N)]
    dim_out = [comb_dim[i][1] for i in range(N)]
    axes_in = [2*j for j in range(N)]
    axes_out = [2*j+1 for j in range(N)]
    for i in range(N):
        H_in = sum(
            extend(H(dim_in[j]), dim_in, j)
            for j in range(i + 1)
        )
        H_out = sum(
            extend(H(dim_out[j]), comb_dim, axes_out[j])
            for j in range(i + 1)
        )
        conds += [
            ptrace(H_out @ T_sup_energy, comb_dim, axes_out) << H_in.T + E * np.eye(H_in.shape[0])
        ]

    # minimize objective
    obj = cp.Minimize(2 - avg_cost)
    prob = cp.Problem(obj, conds)

    return prob, lambda_x_real, lambda_x_imag, E, mul_val, T
    

def calc(args):
    mu = args['mu']
    sigma = args['sigma']
    dim = args['d']

    prob, lambda_x_real, lambda_x_imag, E, mul_val, T = construct_prob(dim, mu, sigma)

    for i in range(dim):
        lambda_x_real[i].value = np.real(args['lambda_x'][i])
        lambda_x_imag[i].value = np.imag(args['lambda_x'][i])
    
    E.value = args['E']

    #print(f'lambda_x: {args["lambda_x"]}, E: {E.value}, sigma: {sigma}')
    prob.solve(warm_start=True, solver=cp.MOSEK, verbose=False)


    sol = {
        'val': prob.value,
        'energy': E.value,
        'sigma': sigma,
        'T': T.value,
        'lambda_x': args['lambda_x'],
    }
    
    return sol

@memory.cache
def construct_C(theta_val=np.pi, is_symbol=False):
    theta = sympy.symbols('theta', real=True)

    Omega = sympy.Matrix([
        [1],
        [0],
        [0],
        [1]
    ])

    U_theta = sympy.Matrix([
        [1, 0],
        [0, sympy.exp(sympy.I * theta)]
    ])

    C_theta = sympy.kronecker_product(sympy.eye(2), U_theta) @ Omega @ Omega.H @ sympy.kronecker_product(sympy.eye(2), U_theta).H

    dC_theta = sympy.diff(C_theta, theta)

    if is_symbol:
        return C_theta, dC_theta
    else:
        C_theta = C_theta.subs(theta, theta_val)
        dC_theta = dC_theta.subs(theta, theta_val)

        return np.array(C_theta, dtype=complex), np.array(dC_theta, dtype=complex)


def solve_accurate():
    E_val = np.linspace(0, 1, 15).tolist()
    lambda_val = np.exp(1j * np.linspace(np.pi/2, 3*np.pi/2, 20))
    sigma_val = [0.5]
    #sigma_val = [0.0001, 0.001, 0.01, 0.1]

    print('calcutating J...')
    for sigma in sigma_val:
        construct_J(np.pi, sigma)
        print(f'J({np.pi}, {sigma}) finished')
    print('J calculation finished')
    
    res_accurate = []

    args = [
        {
            'lambda_x': np.reshape(lambda_x, (2,)).tolist(),
            'E': E,
            'd': 2,
            'mu': np.pi,
            'sigma': sigma,
        }
        for lambda_x in itertools.product(lambda_val, repeat=2)
        for E in E_val
        for sigma in sigma_val
    ]
        
    res_accurate = Parallel(n_jobs=-1)(delayed(calc)(arg) for arg in tqdm(args))
    
    opt_val = {
        (E, sigma): min(
            [sol for sol in res_accurate if sol['energy'] == E and sol['sigma'] == sigma], 
            key=lambda x: x['val']
        )
        for E in E_val
        for sigma in sigma_val
    }

    with open('narrow_distribution.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['E', 'sigma', 'val'])
        for E in E_val:
            for sigma in sigma_val:
                writer.writerow(
                    [
                        E, 
                        sigma, 
                        opt_val[(E, sigma)]['val']
                    ]
                )

    
if __name__ == '__main__':
    solve_accurate()