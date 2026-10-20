"""Condition numbers, the trace bound, and experiments in how many digits a solve loses.

Notebook 4 builds all of these. A float64 carries about 16 significant digits, and a problem
with condition number kappa can lose about log10(kappa) of them.
"""

import itertools

import numpy as np

from .solvers import cholesky, forward_substitution, lstsq_normal, lstsq_qr

EPS = np.finfo(float).eps  # 2.2e-16: the gap between 1 and the next float64


def cond(G):
    """kappa(G) = largest eigenvalue / smallest, for symmetric positive-definite G."""
    eigenvalues = np.linalg.eigvalsh(G)
    return eigenvalues[-1] / eigenvalues[0]


def stretch_factors(R):
    """(smallest, largest) of ||R w|| over unit vectors w: how much R shrinks and stretches.

    These are R's smallest and largest singular values. NumPy finds them from R itself,
    without forming R^T R, so they stay accurate even when R is badly conditioned.
    """
    sigma = np.linalg.svd(R, compute_uv=False)
    return sigma[-1], sigma[0]


def cond_columns(R):
    """kappa(R) = largest stretch / smallest stretch."""
    smallest, largest = stretch_factors(R)
    return largest / smallest


def trace_bound(G):
    """tr(G) tr(G^-1), an upper bound on kappa(G) that needs no eigenvalues.

    tr(G^-1) is the sum of the squares of the entries of L^-1, where G = L L^T, and L^-1 comes
    from m triangular solves, one per column of the identity.
    """
    L = cholesky(G)
    m = len(G)
    L_inverse = np.column_stack([forward_substitution(L, np.eye(m)[:, j]) for j in range(m)])
    return np.trace(G) * np.sum(L_inverse**2)


def support_conds(G):
    """kappa(G_PP) for every non-empty support P, as {support: kappa}."""
    m = len(G)
    return {
        P: cond(G[np.ix_(P, P)])
        for size in range(1, m + 1)
        for P in itertools.combinations(range(m), size)
    }


def digits_lost(relative_error):
    """How many of float64's digits an error costs: log10(error / eps), at least 0."""
    return float(np.log10(max(relative_error, EPS) / EPS))


def unit_pair(theta, n=50, seed=0):
    """An n x 2 matrix of two unit columns `theta` degrees apart, in a random direction of R^n.

    The random rotation keeps the columns from having the tidy entries (mostly zeros) of a
    two-dimensional example, which would let some rounding errors cancel by luck.
    """
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, 2)))
    t = np.radians(theta)
    return Q @ np.array([[1.0, np.cos(t)], [0.0, np.sin(t)]])


def relative_error(solve, R, w_true):
    """The relative error of solve(R, R w_true) against w_true, or inf if the solve fails."""
    with np.errstate(all="ignore"):
        w = solve(R, R @ w_true)
    if not np.all(np.isfinite(w)):
        return np.inf
    return np.linalg.norm(w - w_true) / np.linalg.norm(w_true)


def digits_lost_experiment(R, trials=20, seed=1):
    """(normal equations, QR): the median digits lost fitting s = R w for random w.

    The fit is exact, so the true weights are known, and every digit lost is lost to rounding.
    """
    rng = np.random.default_rng(seed)
    results = {lstsq_normal: [], lstsq_qr: []}
    for _ in range(trials):
        w_true = rng.normal(size=R.shape[1])
        for solve, errors in results.items():
            errors.append(relative_error(solve, R, w_true))
    return tuple(min(digits_lost(np.median(errors)), 16.0) for errors in results.values())
