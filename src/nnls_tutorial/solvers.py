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


def nnls_enumerate_normal(R, s):
    """nnls_enumerate written with the normal equations, as notebook 2 builds it.

    The same algorithm as notebook 2's nnls_by_enumeration: every support is solved from G, c and
    s.s alone. Notebook 3 times it against the active-set method.
    """
    G, c, ss = normal_equations(R, s)
    m = len(c)
    best_w, best_error = np.zeros(m), ss
    for support in supports(m):
        P = list(support)
        if np.linalg.matrix_rank(G[np.ix_(P, P)]) < len(P):
            continue
        z, error = support_fit(G, c, ss, support)
        if np.all(z >= 0) and error < best_error:
            best_w, best_error = np.zeros(m), error
            best_w[P] = z
    return best_w


@dataclass
class Step:
    """One step of the active-set method, recorded so that a run can be replayed and drawn."""

    kind: str  # "start", "add", "fit", "backtrack" or "stop"
    w: np.ndarray  # the weights after this step
    support: tuple  # the switched-on positions P after this step, in increasing order
    changed: tuple  # the position added ("add") or the positions dropped ("backtrack")
    target: np.ndarray  # z, the least-squares fit on P that this step aimed at, or None
    gradient: np.ndarray  # lambda = grad f(w), at the weights after this step
    error: float  # f(w), at the weights after this step


def active_set_steps(R, s):
    """Lawson and Hanson's active-set method for NNLS, recorded one step at a time.

    Start at w = 0 with nothing switched on. Then repeat:
      - stop if no switched-off weight has a negative slope lambda_i (the KKT conditions hold);
      - otherwise switch on the weight whose slope is most negative ("add"), and
      - fit by least squares on the switched-on weights. If that fit is allowed, move to it
        ("fit"). If not, move toward it only as far as the quadrant allows, switch off the
        weights that reach zero, and fit again ("backtrack").
    Like notebook 2's enumeration, every fit solves G_PP z = c_P. Notebook 3 proves that the
    method stops, and at the NNLS answer. The last step's w is the answer.
    """
    G, c, ss = normal_equations(R, s)
    m = len(c)
    # slopes this close to zero are zero, up to rounding
    tol = 10 * m * np.finfo(float).eps * max(np.abs(G).max(), np.abs(c).max(), 1e-300)
    w = np.zeros(m)
    P = []
    steps = []

    def record(kind, changed=(), target=None):
        lam = 2 * (G @ w - c)
        steps.append(
            Step(kind, w.copy(), tuple(sorted(P)), changed, target, lam, objective(R, s, w))
        )
        return lam

    lam = record("start")
    for _ in range(3 * m + 10):
        off = [i for i in range(m) if i not in P]
        if not off or min(lam[i] for i in off) >= -tol:
            record("stop")
            return steps
        t = min(off, key=lambda i: lam[i])  # the steepest way down that the fence allows
        P.append(t)
        record("add", (t,))
        while True:
            z = np.zeros(m)
            z[P] = support_fit(G, c, ss, P)[0]
            if z[t] <= 0 and np.all(w[P] == 0):
                # Theorem 13 says z_t > 0; only rounding can do this, when lambda_t is zero to
                # working precision. Treat it as zero.
                P.remove(t)
                record("stop")
                return steps
            if np.all(z[P] > 0):
                w = z
                lam = record("fit", target=z)
                break
            # move from w toward z until the first weight reaches zero, and switch it off
            blocking = [i for i in P if z[i] <= 0]
            alpha, first = min((w[i] / (w[i] - z[i]), i) for i in blocking)
            w = w + alpha * (z - w)
            dropped = tuple(i for i in P if i == first or w[i] <= tol)
            w[list(dropped)] = 0
            P = [i for i in P if i not in dropped]
            lam = record("backtrack", dropped, target=z)
    raise RuntimeError("the active-set method did not stop; the problem may be badly conditioned")


def nnls_active_set(R, s):
    """NNLS by Lawson and Hanson's active-set method: the last step of active_set_steps."""
    return active_set_steps(R, s)[-1].w


def cholesky(G):
    """The lower-triangular L with G = L L^T, for symmetric positive-definite G.

    Computed column by column, as notebook 4 derives it: each column's diagonal entry first,
    then the entries below it. Each entry uses only entries of L that are already known.
    """
    m = G.shape[0]
    L = np.zeros_like(G, dtype=float)
    for j in range(m):
        # what's left of G_jj after the columns already factored; > 0 when G is positive definite
        L[j, j] = np.sqrt(G[j, j] - L[j, :j] @ L[j, :j])
        for i in range(j + 1, m):
            L[i, j] = (G[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L


def forward_substitution(L, c):
    """Solve L y = c for lower-triangular L, first row first."""
    y = np.zeros(len(c))
    for i in range(len(c)):
        y[i] = (c[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_substitution(U, y):
    """Solve U z = y for upper-triangular U, last row first."""
    z = np.zeros(len(y))
    for i in reversed(range(len(y))):
        z[i] = (y[i] - U[i, i + 1 :] @ z[i + 1 :]) / U[i, i]
    return z


def solve_cholesky(G, c):
    """Solve G z = c for symmetric positive-definite G: factor G = L L^T, then two triangular
    solves, L y = c forward and L^T z = y backward."""
    L = cholesky(G)
    return back_substitution(L.T, forward_substitution(L, c))


def lstsq_normal(R, s):
    """Least squares through the normal equations R^T R w = R^T s, solved by Cholesky."""
    return solve_cholesky(R.T @ R, R.T @ s)


def lstsq_qr(R, s):
    """Least squares through the QR factorization R = Q U, solving U w = Q^T s.

    Q has orthonormal columns and U is upper triangular. (The usual name is QR, but R is
    already taken.) R^T R is never formed, so its condition number is never squared.
    """
    Q, U = np.linalg.qr(R)
    return back_substitution(U, Q.T @ s)
