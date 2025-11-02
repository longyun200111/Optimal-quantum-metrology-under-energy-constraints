import numpy as np

def integrate(
    func,
    a,
    b,
    tol_abs: float = 1e-10,
    tol_rel: float = 1e-10,
    max_k: int = 12,
    min_k: int = 3,
    eps: float = 1e-6,
    **kwargs,
):
    """
    Adaptive Romberg integration (adaptive step size refinement) for matrices, supporting complex numbers and infinite intervals.

    - Uses the Romberg method (based on the trapezoidal rule and Richardson extrapolation) and automatically refines the sampling points.
    - Finite, semi-infinite, and doubly infinite intervals are unified into a bounded interval via variable substitution, and then Romberg is applied with respect to `u`.
    - The function `func: t -> matrix (m,n)` can return a complex matrix; the return value has the same shape as the integrand matrix.

    Parameters:
    - func: The integrand function, with signature `func(t) -> np.ndarray(m,n)`, where elements can be complex.
    - a, b: The lower and upper limits of integration; can be finite values, `-np.inf`, or `np.inf`.
    - tol_abs, tol_rel: Absolute and relative convergence thresholds (for the Frobenius norm).
    - max_k: Maximum Romberg level (number of grid points N=2**k+1).
    - min_k: Initial Romberg level (recommended >=3).
    - eps: Endpoint avoidance amount when mapping to a bounded interval (to avoid singularities).

    Returns:
    - The resulting integral matrix (complex), with the same shape as the return value of `func(t)`.
    """

    # Shape detection: perform one call with a finite t to determine the output shape
    if np.isfinite(a):
        probe_t = a
    elif np.isfinite(b):
        probe_t = b
    else:
        probe_t = 0.0
    sample = np.asarray(func(probe_t), dtype=complex)
    out_shape = sample.shape

    # Map (a,b) to a bounded u interval and a weight w(u), such that ∫ f(t) dt = ∫ f(t(u)) w(u) du
    if np.isfinite(a) and np.isfinite(b):
        # Finite interval: u=t, w=1, [u_a,u_b]=[a,b]
        def t_of_u(u):
            return u
        def w_of_u(u):
            return np.ones_like(u)
        u_a, u_b = float(a), float(b)
    elif np.isfinite(a) and np.isinf(b):
        # Semi-infinite [a, +inf): t = a + u/(1-u), u ∈ (0,1)
        def t_of_u(u):
            return a + u / (1.0 - u)
        def w_of_u(u):
            return 1.0 / (1.0 - u) ** 2
        u_a, u_b = eps, 1.0 - eps
    elif np.isinf(a) and np.isfinite(b):
        # Semi-infinite (-inf, b]: t = b - u/(1-u), u ∈ (0,1)
        def t_of_u(u):
            return b - u / (1.0 - u)
        def w_of_u(u):
            return 1.0 / (1.0 - u) ** 2
        u_a, u_b = eps, 1.0 - eps
    elif np.isinf(a) and np.isinf(b):
        # Doubly infinite: t = tan(pi u / 2), u ∈ (-1,1)
        def t_of_u(u):
            return np.tan(0.5 * np.pi * u)
        def w_of_u(u):
            return 0.5 * np.pi / (np.cos(0.5 * np.pi * u) ** 2)
        u_a, u_b = -1.0 + eps, 1.0 - eps
    else:
        raise ValueError("Invalid bounds configuration for integration.")

    def f_u(u):
        # u can be a scalar or an array; for batch evaluation, process as an array pipeline
        uu = np.asarray(u, dtype=float)
        tt = t_of_u(uu)
        ww = w_of_u(uu)
        vals = np.array([wi * np.asarray(func(ti), dtype=complex) for wi, ti in zip(np.ravel(ww), np.ravel(tt))])
        return vals

    # Romberg adaptive quadrature: refine the grid, construct trapezoidal values, and perform Richardson extrapolation
    prev_row = None  # Save the previous Romberg row (list: R[m, j], j=0..J)
    result = None
    for k in range(min_k, max_k + 1):
        N = 2 ** k + 1
        u = np.linspace(u_a, u_b, N)
        fu = f_u(u)  # Shape (N, *out_shape)
        dx = (u_b - u_a) / (N - 1)
        # Trapezoidal rule T_k
        Tk = dx * (0.5 * fu[0] + fu[1:-1].sum(axis=0) + 0.5 * fu[-1])

        # Construct the current Romberg row
        row = [Tk]
        if prev_row is not None:
            # The maximum order of J increases with k
            max_j = len(prev_row)
            for j in range(1, max_j + 1):
                Rj = row[j - 1] + (row[j - 1] - prev_row[j - 1]) / (4 ** j - 1)
                row.append(Rj)

            # Convergence criterion: difference between the current highest order and the previous highest order
            curr_top = row[-1]
            prev_top = prev_row[-1]
            diff_norm = np.linalg.norm(curr_top - prev_top)
            base_norm = np.linalg.norm(curr_top)
            if diff_norm <= tol_abs + tol_rel * max(base_norm, 1.0):
                result = curr_top
                break

        prev_row = row
        result = row[-1]

    return np.asarray(result, dtype=complex)
