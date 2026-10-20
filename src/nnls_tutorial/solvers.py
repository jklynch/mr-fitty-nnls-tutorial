"""Least squares and non-negative least squares, written to be read.

Notation throughout the series: R is the matrix whose columns r_1, ..., r_m are the reference
spectra, s is the measured spectrum, and w holds the weights, so the fit is R w and the
residual is e = s - R w.

These favor clarity over speed. Notebook 5 builds the fast versions; every solver here is
checked against scipy.optimize.nnls in tests/.
"""

import itertools
from dataclasses import dataclass

import numpy as np


def lstsq(R, s):
    """The unconstrained least-squares weights: minimize ||R w - s||, any sign allowed."""
    w, *_ = np.linalg.lstsq(R, s, rcond=None)
    return w


def supports(m):
    """Every non-empty subset of the m columns, smallest first, as tuples of positions."""
    return [
        support for size in range(1, m + 1) for support in itertools.combinations(range(m), size)
    ]


@dataclass
class Candidate:
    """The least-squares fit using only the columns in `support`, zero weight on the rest."""

    support: tuple
    w: np.ndarray  # (m,) full-width weights, None if the support's columns are dependent
    error: float  # ||R w - s||^2, inf if w is None
    feasible: bool  # w exists and every weight is >= 0


def candidates(R, s):
    """One candidate per support, plus the empty support (w = 0) first.

    The NNLS answer is always one of these: notebook 2 proves it. Supports whose columns are
    linearly dependent have no unique fit, and are reported with w = None.
    """
    m = R.shape[1]
    found = [Candidate((), np.zeros(m), float(s @ s), True)]
    for support in supports(m):
        R_P = R[:, support]
        if np.linalg.matrix_rank(R_P) < len(support):
            found.append(Candidate(support, None, np.inf, False))
            continue
        w = np.zeros(m)
        w[list(support)] = lstsq(R_P, s)
        found.append(Candidate(support, w, float(np.sum((R @ w - s) ** 2)), bool(np.all(w >= 0))))
    return found


def nnls_enumerate(R, s):
    """NNLS by trying every support and keeping the best feasible fit.

    The work grows as 2**m, so this is for small m. Ties go to the smaller support.
    """
    feasible = [c for c in candidates(R, s) if c.feasible]
    return min(feasible, key=lambda c: c.error).w


def nnls_enumerate_many(R, S):
    """nnls_enumerate for many targets at once, all sharing R: S is (k, n), one target per row.

    Returns (W, winner): W is (k, m), and winner[i] is the position of row i's winning support
    in [()] + supports(m), so 0 means w = 0. With R fixed, each support's least-squares weights
    are a fixed linear map of s, so every target is solved by one matrix product per support.
    Supports with linearly dependent columns are skipped. Used for the region maps.
    """
    S = np.atleast_2d(S)
    k, m = S.shape[0], R.shape[1]
    all_supports = [()] + supports(m)
    errors = np.full((k, len(all_supports)), np.inf)
    errors[:, 0] = np.sum(S * S, axis=1)
    weights = [np.zeros((k, 0))]
    for i, support in enumerate(all_supports[1:], start=1):
        R_P = R[:, support]
        if np.linalg.matrix_rank(R_P) < len(support):
            weights.append(None)
            continue
        Z = S @ np.linalg.pinv(R_P).T  # (k, |P|) least-squares weights for every target
        weights.append(Z)
        # ||R_P z - s||^2 = s.s - z.(R_P^T s) at the least-squares z; notebook 2 proves it
        error = errors[:, 0] - np.sum(Z * (S @ R_P), axis=1)
        errors[:, i] = np.where(np.all(Z >= 0, axis=1), error, np.inf)
    winner = np.argmin(errors, axis=1)  # first of equal minima: the smaller support
    W = np.zeros((k, m))
    for i, support in enumerate(all_supports[1:], start=1):
        rows = winner == i
        if np.any(rows):
            W[np.ix_(rows, support)] = weights[i][rows]
    return W, winner


def support_of(w, tol=1e-12):
    """The positions of the weights that are switched on."""
    return tuple(int(i) for i in np.flatnonzero(w > tol))


def objective(R, s, w):
    """f(w) = ||R w - s||^2, the quantity least squares and NNLS minimize."""
    return float(np.sum((R @ w - s) ** 2))


def gradient(R, s, w):
    """The gradient of f(w) = ||R w - s||^2, which is 2 R^T (R w - s) = -2 R^T e."""
    return 2 * R.T @ (R @ w - s)


def normal_equations(R, s):
    """(G, c, ss) = (R^T R, R^T s, s^T s): everything f depends on, in m x m numbers.

    f(w) = w^T G w - 2 c^T w + ss, so once these are computed, no n-long vector is needed
    again: every candidate's weights and error come from G, c and ss alone.
    """
    return R.T @ R, R.T @ s, float(s @ s)


def support_fit(G, c, ss, support):
    """(z, error): least squares on the columns in `support`, from the normal equations alone.

    z solves G_PP z = c_P, and the error uses the shortcut ||R_P z - s||^2 = ss - z . c_P,
    which holds at that z (notebook 2 proves it). The support's columns must be independent.
    """
    P = list(support)
    z = np.linalg.solve(G[np.ix_(P, P)], c[P])
    return z, ss - z @ c[P]
