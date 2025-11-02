'''
Calculate the optimal precision for the local battery
with phase channels e^{-iθZ/2} and e^{iθZ/2} 
preceded by bit-flip channel ρ → 0.5 * ρ+ 0.5 * X ρ X
'''

import cvxpy as cp
import scipy.linalg
import itertools
import matplotlib.pyplot as plt
from constants import get_H
import numpy as np
import scipy.sparse as sp
import sympy
from functools import cache
import csv
from tqdm import tqdm
import multiprocessing as mp

from utils import *

def pdf(x):
    return 1 / (2*np.pi)
    # mu = 1
    # sigma = 0.1
    # return 1 / (sigma * sympy.sqrt(2 * np.pi)) * sympy.exp(-0.5 * ((x - mu) / sigma) ** 2)

@cache
def construct_J():
    theta = sympy.symbols('theta')

    Omega = sympy.Matrix([
        [1],
        [0],
        [0],
        [1]
    ])

    Z = sympy.Matrix([
        [1, 0],
        [0, -1]
    ])

    X = sympy.Matrix([
        [0, 1],
        [1, 0]
    ])

    U_theta = sympy.exp(-0.5j * theta * Z)

    Kraus1 = [
        0.5 * sympy.eye(2),
        0.5 * sympy.Matrix([
            [0, 1],
            [1, 0]
        ])
    ]
    
    result_matrix1 = sympy.zeros(4, 4)
    for K in Kraus1:
        result_matrix1 += sympy.kronecker_product(sympy.eye(2), K) @ Omega @ Omega.H @ sympy.kronecker_product(sympy.eye(2), K).H

    C_theta_1 = sympy.kronecker_product(sympy.eye(2), U_theta) @ result_matrix1 @ sympy.kronecker_product(sympy.eye(2), U_theta).H

    V_theta = sympy.exp(0.5j * theta * Z)
    Kraus2 = [
        0.5 * sympy.eye(2),
        0.5 * sympy.Matrix([
            [0, 1],
            [1, 0]
        ])
    ]

    result_matrix2 = sympy.zeros(4, 4)
    for K in Kraus2:
        result_matrix2 += sympy.kronecker_product(sympy.eye(2), K) @ Omega @ Omega.H @ sympy.kronecker_product(sympy.eye(2), K).H

    C_theta_2 = sympy.kronecker_product(sympy.eye(2), V_theta) @ result_matrix2 @ sympy.kronecker_product(sympy.eye(2), V_theta).H

    C_theta = sympy.kronecker_product(C_theta_1, C_theta_2)

    J = sympy.integrate(sympy.exp(-1j * theta) * C_theta.T * pdf(theta), (theta, 0, 2 * np.pi))
    return np.array(J, dtype=complex)

@cache
def construct_prob(p=1, dim = 2):
    avg_cost = 0
    conds = []

    p = [p, 1 - p]
    lambda_x_real = [cp.Parameter() for _ in range(2)]
    lambda_x_imag = [cp.Parameter() for _ in range(2)]
    E = cp.Parameter() 

    comb_dim = [(1, 2), (2, 2), (2, 2)]

    mul_val = [None for _ in range(2)]

    N = len(comb_dim)

    T = cp.Variable((np.prod(comb_dim), np.prod(comb_dim)), hermitian=True)
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
    J = construct_J()
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
    En = [cp.Variable() for _ in range(N)]
    for i in range(N):
        H_in = extend(H(dim_in[i]), dim_in, i)
        H_out = extend(H(dim_out[i]), comb_dim, axes_out[i])
        conds += [
            ptrace(H_out @ T_sup_energy, comb_dim, axes_out) << H_in.T + En[i] * np.eye(H_in.shape[0])
        ]
    conds += [
        En[i] >= 0
        for i in range(N)
    ]
    conds += [
        cp.sum(En) <= E
    ]

    # minimize objective
    obj = cp.Minimize(2 - avg_cost)
    prob = cp.Problem(obj, conds)

    return prob, lambda_x_real, lambda_x_imag, E, mul_val, T
    

def calc(args, dim=2):
    p = 1

    prob, lambda_x_real, lambda_x_imag, E, mul_val, T_sup = construct_prob(p, dim)

    for i in range(2):
        lambda_x_real[i].value = np.real(args['lambda_x'][i])
        lambda_x_imag[i].value = np.imag(args['lambda_x'][i])
    
    E.value = args['E']

    # for _ in range(10):
    prob.solve(warm_start=True, solver=cp.MOSEK)


    sol = {
        'val': prob.value,
        'energy': E.value,
        'p': [p, 1 - p]
    }

    #print(f'energy: {E.value}')

    #sol['Tx'] = [Tx[i].value for i in range(dim)]
    #sol['rho'] = rho.value
    
    return sol

def solve_accurate():
    E_val = sorted([0.0125, 0.025] + np.linspace(0, 0.5, 5).tolist())
    lambda_val = np.exp(1j * np.linspace(0, 2*np.pi, 20))
    
    res_accurate = []

    args = [
        {
            'lambda_x': np.reshape(lambda_x, (2,)).tolist(),
            'E': E
        }
        for lambda_x in itertools.product(lambda_val, repeat=2)
        for E in E_val
    ]
        
    pbar = tqdm(total=len(args))
    pool = mp.Pool()
    update = lambda *args: pbar.update()
    task = []
    for arg in args:
        task.append(
            pool.apply_async(calc, args=(arg,), callback=update)
        )
    for t in task:
        res_accurate.append(t.get())
    pool.close()
    pool.join()
    
    opt_val_causal = {
        E: min([sol['val'] for sol in res_accurate if sol['energy'] == E])
        for E in E_val
    }

    with open('def_loc.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['E', 'val'])
        for E in E_val:
            writer.writerow([E, opt_val_causal[E]])

if __name__ == '__main__':
    solve_accurate()