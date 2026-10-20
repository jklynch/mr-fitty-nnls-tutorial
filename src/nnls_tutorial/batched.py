"""Many small NNLS fits at once: a SciPy loop, batched NumPy, and compiled Numba.

Every fit here shares the references R and the target s, and differs only in which energies it
is fitted on: fit j uses the rows where train[j] is True. That's the shape of the refits in
MrFitty's blocked cross-validation, where each draw holds out a few blocks of energies. The fit
is NNLS by trying every support (notebook 2), each support solved by Cholesky (notebook 4).
"""

import itertools

import numpy as np
import scipy.optimize
from numba import njit, prange

from .solvers import supports

CONDITION_LIMIT = 1e8  # fits whose G is worse than this go to SciPy (notebook 4)


# ---------------------------------------------------------------------------------------------
# The draws


def holdout_masks(n, draws, blocks=4, block_length=10, seed=0):
    """(draws, n) bools, True where a draw holds the energy out: `blocks` random runs each."""
    rng = np.random.default_rng(seed)
    held_out = np.zeros((draws, n), dtype=bool)
    starts = rng.integers(0, n - block_length + 1, size=(draws, blocks))
    for j in range(draws):
        for start in starts[j]:
            held_out[j, start : start + block_length] = True
    return held_out


def prediction_errors(R, s, W, held_out):
    """For each draw, the root-mean-square error of its fit R w on its held-out energies."""
    residual = W @ R.T - s
    return np.sqrt(np.sum(held_out * residual**2, axis=1) / held_out.sum(axis=1))


# ---------------------------------------------------------------------------------------------
# Version 0: one SciPy call per draw


def nnls_scipy_loop(R, s, train):
    """W[j] = scipy.optimize.nnls on draw j's training rows."""
    W = np.zeros((len(train), R.shape[1]))
    for j, rows in enumerate(train):
        W[j], _ = scipy.optimize.nnls(R[rows], s[rows])
    return W


# ---------------------------------------------------------------------------------------------
# Version 1: batched NumPy


def gram_einsum(R, s, train):
    """(G, c, ss) for every draw at once, by np.einsum: G[j] = R^T diag(train[j]) R."""
    T = train.astype(float)
    G = np.einsum("jn,na,nb->jab", T, R, R)
    return G, (T * s) @ R, T @ (s * s)


def gram_blas(R, s, train):
    """The same as gram_einsum, as one matrix product, which NumPy hands to BLAS.

    G[j, a, b] = sum over n of T[j, n] R[n, a] R[n, b], so if products[n] is the m x m table
    of R[n, a] R[n, b], flattened to m^2 numbers, then G is T @ products, reshaped.
    """
    T = train.astype(float)
    n, m = R.shape
    products = (R[:, :, None] * R[:, None, :]).reshape(n, m * m)
    return (T @ products).reshape(len(T), m, m), (T * s) @ R, T @ (s * s)


def batched_cholesky_solve(G, c):
    """Solve G z = c for a whole stack of small positive-definite systems at once.

    G is (..., p, p) and c is (..., p). This is solvers.cholesky and the two triangular solves,
    with every step one array operation across the whole stack: the loops run over p, not
    over the systems, so there are about p^2 operations however many systems there are.
    """
    p = G.shape[-1]
    L = np.zeros_like(G)
    for j in range(p):
        L[..., j, j] = np.sqrt(G[..., j, j] - np.sum(L[..., j, :j] ** 2, axis=-1))
        for i in range(j + 1, p):
            L[..., i, j] = (G[..., i, j] - np.sum(L[..., i, :j] * L[..., j, :j], axis=-1)) / L[
                ..., j, j
            ]
    y = np.zeros_like(c)
    for i in range(p):
        y[..., i] = (c[..., i] - np.sum(L[..., i, :i] * y[..., :i], axis=-1)) / L[..., i, i]
    z = np.zeros_like(c)
    for i in reversed(range(p)):
        z[..., i] = (y[..., i] - np.sum(L[..., i + 1 :, i] * z[..., i + 1 :], axis=-1)) / L[
            ..., i, i
        ]
    return z


def batched_nnls(R, s, train, condition_limit=CONDITION_LIMIT):
    """NNLS for every draw at once, by trying every support (notebook 2), in NumPy.

    Returns W, (draws, m). Draws whose G is worse conditioned than `condition_limit` are
    refitted one at a time by SciPy (notebook 4).
    """
    G, c, ss = gram_blas(R, s, train)
    k, m = c.shape
    eigenvalues = np.linalg.eigvalsh(G)  # one check per draw covers every support
    careful = ~(eigenvalues[:, 0] > eigenvalues[:, -1] / condition_limit)
    fast = np.flatnonzero(~careful)
    G, c, ss = G[fast], c[fast], ss[fast]

    # one column of errors per support, the empty support first; solved a size at a time
    errors = [ss[:, None]]
    solved = []
    for size in range(1, m + 1):
        P = np.array(list(itertools.combinations(range(m), size)))  # (sets, size)
        G_P = G[:, P[:, :, None], P[:, None, :]]  # (draws, sets, size, size)
        c_P = c[:, P]  # (draws, sets, size)
        Z = batched_cholesky_solve(G_P, c_P)
        error = ss[:, None] - np.sum(Z * c_P, axis=-1)  # notebook 2's shortcut
        error[np.any(Z < 0, axis=-1)] = np.inf  # not allowed
        errors.append(error)
        solved.append((P, Z))
    winner = np.argmin(np.concatenate(errors, axis=1), axis=1)  # ties go to smaller supports

    # copy out only the winners' weights, a size at a time
    W = np.zeros((k, m))
    first = 1
    for P, Z in solved:
        mine = np.flatnonzero((winner >= first) & (winner < first + len(P)))
        chosen = winner[mine] - first
        W[fast[mine, None], P[chosen]] = Z[mine, chosen]
        first += len(P)
    for j in np.flatnonzero(careful):
        W[j], _ = scipy.optimize.nnls(R[train[j]], s[train[j]])
    return W


# ---------------------------------------------------------------------------------------------
# Version 2: compiled with Numba


def support_masks(m):
    """Every non-empty support of m columns as a bitmask, smallest supports first."""
    return np.array([sum(1 << i for i in P) for P in supports(m)], dtype=np.int64)


@njit(cache=True)
def _refit_one(R, s, rows, masks, condition_limit, w):
    """NNLS on one draw's training rows, by trying every support. Writes the weights into w.

    Returns False, leaving w alone, if the draw's G fails the trace-bound check (notebook 4,
    Theorem 22), so the caller can hand it to SciPy. Plain loops throughout: compiled, they're
    fast, and nothing here calls LAPACK.
    """
    n, m = R.shape
    G = np.zeros((m, m))
    c = np.zeros(m)
    ss = 0.0
    for r in range(n):
        if rows[r]:
            for a in range(m):
                c[a] += R[r, a] * s[r]
                for b in range(m):
                    G[a, b] += R[r, a] * R[r, b]
            ss += s[r] * s[r]

    # the check: factor all of G, then tr(G) * ||L^-1||_F^2 <= limit
    L = np.zeros((m, m))
    for j in range(m):
        d = G[j, j]
        for k in range(j):
            d -= L[j, k] * L[j, k]
        if d <= 0.0:
            return False
        L[j, j] = np.sqrt(d)
        for i in range(j + 1, m):
            x = G[i, j]
            for k in range(j):
                x -= L[i, k] * L[j, k]
            L[i, j] = x / L[j, j]
    trace_G = 0.0
    sum_squares = 0.0
    e = np.zeros(m)
    for col in range(m):  # column `col` of L^-1, by forward substitution on e_col
        trace_G += G[col, col]
        for i in range(m):
            x = 1.0 if i == col else 0.0
            for k in range(i):
                x -= L[i, k] * e[k]
            e[i] = x / L[i, i]
            sum_squares += e[i] * e[i]
    if trace_G * sum_squares > condition_limit:
        return False

    # every support; G passed the check, so every G_PP is positive definite (Theorem 21)
    best_error = ss
    for q in range(m):
        w[q] = 0.0
    P = np.zeros(m, dtype=np.int64)
    LP = np.zeros((m, m))
    y = np.zeros(m)
    z = np.zeros(m)
    for mask in masks:
        p = 0
        for q in range(m):
            if (mask >> q) & 1:
                P[p] = q
                p += 1
        for j in range(p):  # Cholesky of G_PP
            d = G[P[j], P[j]]
            for k in range(j):
                d -= LP[j, k] * LP[j, k]
            LP[j, j] = np.sqrt(d)
            for i in range(j + 1, p):
                x = G[P[i], P[j]]
                for k in range(j):
                    x -= LP[i, k] * LP[j, k]
                LP[i, j] = x / LP[j, j]
        for i in range(p):  # L y = c_P
            x = c[P[i]]
            for k in range(i):
                x -= LP[i, k] * y[k]
            y[i] = x / LP[i, i]
        allowed = True
        for i in range(p - 1, -1, -1):  # L^T z = y
            x = y[i]
            for k in range(i + 1, p):
                x -= LP[k, i] * z[k]
            z[i] = x / LP[i, i]
            if z[i] < 0.0:
                allowed = False
        if not allowed:
            continue
        error = ss
        for i in range(p):
            error -= z[i] * c[P[i]]
        if error < best_error:
            best_error = error
            for q in range(m):
                w[q] = 0.0
            for i in range(p):
                w[P[i]] = z[i]
    return True


@njit(parallel=True, cache=True)
def _refit_draws(R, s, train, masks, condition_limit):
    k = train.shape[0]
    W = np.zeros((k, R.shape[1]))
    passed = np.zeros(k, dtype=np.bool_)
    for j in prange(k):
        passed[j] = _refit_one(R, s, train[j], masks, condition_limit, W[j])
    return W, passed


def compiled_nnls(R, s, train, condition_limit=CONDITION_LIMIT):
    """NNLS for every draw, compiled: one draw per thread, every support, Cholesky by hand."""
    R = np.ascontiguousarray(R, dtype=float)
    W, passed = _refit_draws(R, s, train, support_masks(R.shape[1]), condition_limit)
    for j in np.flatnonzero(~passed):
        W[j], _ = scipy.optimize.nnls(R[train[j]], s[train[j]])
    return W


@njit(parallel=True, cache=True)
def _condition_checks_lapack(R, train):
    """kappa(G) for every draw by np.linalg.eigvalsh, one draw per thread: notebook 5's
    example of what not to do. Each call goes to LAPACK, which allocates its own memory."""
    k = train.shape[0]
    m = R.shape[1]
    kappa = np.zeros(k)
    for j in prange(k):
        G = np.zeros((m, m))
        for r in range(R.shape[0]):
            if train[j, r]:
                for a in range(m):
                    for b in range(m):
                        G[a, b] += R[r, a] * R[r, b]
        mu = np.linalg.eigvalsh(G)
        kappa[j] = mu[-1] / mu[0]
    return kappa


@njit(parallel=True, cache=True)
def _condition_checks_trace(R, train):
    """The trace bound for every draw, by hand, one draw per thread: no LAPACK."""
    k = train.shape[0]
    m = R.shape[1]
    bound = np.zeros(k)
    for j in prange(k):
        G = np.zeros((m, m))
        for r in range(R.shape[0]):
            if train[j, r]:
                for a in range(m):
                    for b in range(m):
                        G[a, b] += R[r, a] * R[r, b]
        L = np.zeros((m, m))
        for jj in range(m):
            d = G[jj, jj]
            for kk in range(jj):
                d -= L[jj, kk] * L[jj, kk]
            L[jj, jj] = np.sqrt(d)
            for i in range(jj + 1, m):
                x = G[i, jj]
                for kk in range(jj):
                    x -= L[i, kk] * L[jj, kk]
                L[i, jj] = x / L[jj, jj]
        e = np.zeros(m)
        trace_G = 0.0
        sum_squares = 0.0
        for col in range(m):
            trace_G += G[col, col]
            for i in range(m):
                x = 1.0 if i == col else 0.0
                for kk in range(i):
                    x -= L[i, kk] * e[kk]
                e[i] = x / L[i, i]
                sum_squares += e[i] * e[i]
        bound[j] = trace_G * sum_squares
    return bound


def condition_checks(R, train, method):
    """kappa(G) (method="eigvalsh") or its trace bound (method="trace") for every draw."""
    R = np.ascontiguousarray(R, dtype=float)
    if method == "eigvalsh":
        return _condition_checks_lapack(R, train)
    return _condition_checks_trace(R, train)


# ---------------------------------------------------------------------------------------------
# A whole search


def combinations_up_to(pool, largest):
    """Every combination of 1 to `largest` of the pool's columns, smallest first."""
    return [P for size in range(1, largest + 1) for P in itertools.combinations(range(pool), size)]


@njit(parallel=True, cache=True)
def _search_compiled(R_pool, s, held_out, combos, sizes, masks, condition_limit):
    """Every combination on every draw, one combination per thread, the draws in turn.

    combos is (C, largest), padded with -1. Returns W (C, draws, largest), each draw's
    prediction error, and whether it passed the condition check.
    """
    C, largest = combos.shape
    k, n = held_out.shape
    W = np.zeros((C, k, largest))
    pe = np.zeros((C, k))
    passed = np.zeros((C, k), dtype=np.bool_)
    for ci in prange(C):
        p = sizes[ci]
        R = np.empty((n, p))
        for i in range(p):
            R[:, i] = R_pool[:, combos[ci, i]]
        mine = masks[masks < (1 << p)]  # the supports of p columns, still smallest first
        w = np.zeros(p)
        for j in range(k):
            train = ~held_out[j]
            passed[ci, j] = _refit_one(R, s, train, mine, condition_limit, w)
            if passed[ci, j]:
                total = 0.0
                count = 0
                for r in range(n):
                    if held_out[j, r]:
                        x = -s[r]
                        for i in range(p):
                            x += R[r, i] * w[i]
                        total += x * x
                        count += 1
                pe[ci, j] = np.sqrt(total / count)
                for i in range(p):
                    W[ci, j, i] = w[i]
    return W, pe, passed


def search_compiled(spectra, held_out, largest=3, condition_limit=CONDITION_LIMIT):
    """search(), with the whole search compiled: the threads share out the combinations.

    Draws that fail the condition check are refitted by SciPy afterwards.
    """
    combos = combinations_up_to(len(spectra.names), largest)
    padded = np.full((len(combos), largest), -1, dtype=np.int64)
    for ci, P in enumerate(combos):
        padded[ci, : len(P)] = P
    sizes = np.array([len(P) for P in combos], dtype=np.int64)
    R_pool = np.ascontiguousarray(spectra.R, dtype=float)
    W, pe, passed = _search_compiled(
        R_pool, spectra.s, held_out, padded, sizes, support_masks(largest), condition_limit
    )
    results = {}
    for ci, P in enumerate(combos):
        W_P = W[ci, :, : len(P)]
        failed = np.flatnonzero(~passed[ci])
        if len(failed):
            R = spectra.R[:, list(P)]
            for j in failed:
                W_P[j], _ = scipy.optimize.nnls(R[~held_out[j]], spectra.s[~held_out[j]])
            pe[ci, failed] = prediction_errors(R, spectra.s, W_P[failed], held_out[failed])
        results[P] = (W_P, pe[ci])
    return results


def search(spectra, held_out, largest=3, solve=compiled_nnls):
    """Every combination of up to `largest` references, refitted on every draw.

    Returns {combination: (W, prediction errors)}, W being (draws, |combination|).
    """
    train = ~held_out
    results = {}
    for P in combinations_up_to(len(spectra.names), largest):
        R = spectra.R[:, list(P)]
        W = solve(R, spectra.s, train)
        results[P] = (W, prediction_errors(R, spectra.s, W, held_out))
    return results
