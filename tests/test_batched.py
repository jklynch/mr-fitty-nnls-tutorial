"""The batched and compiled solvers against one SciPy call per fit."""

import numpy as np
from problems import random_problem

from nnls_tutorial import batched as B
from nnls_tutorial.data import load_arsenic


def awkward_problem(seed, n=60, draws=40):
    """A random problem from problems.py, with random holdouts."""
    R, s = random_problem(np.random.default_rng(seed), n=n)
    train = ~B.holdout_masks(n, draws, blocks=2, block_length=8, seed=seed)
    return R, s, train


def test_batched_and_compiled_match_scipy():
    for seed in range(40):
        R, s, train = awkward_problem(seed)
        W = B.nnls_scipy_loop(R, s, train)
        message = f"seed {seed}"
        np.testing.assert_allclose(B.batched_nnls(R, s, train), W, atol=1e-9, err_msg=message)
        np.testing.assert_allclose(B.compiled_nnls(R, s, train), W, atol=1e-9, err_msg=message)


def test_gram_matrices_agree():
    R, s, train = awkward_problem(1)
    for a, b in zip(B.gram_einsum(R, s, train), B.gram_blas(R, s, train), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_batched_cholesky_matches_numpy():
    rng = np.random.default_rng(2)
    A = rng.random((7, 20, 4))
    G = np.swapaxes(A, -1, -2) @ A
    c = rng.random((7, 4))
    expected = np.linalg.solve(G, c[..., None])[..., 0]
    np.testing.assert_allclose(B.batched_cholesky_solve(G, c), expected, atol=1e-10)


def test_condition_checks_bound_the_eigenvalue_ratio():
    R, s, train = awkward_problem(3)
    kappa = B.condition_checks(R, train, "eigvalsh")
    bound = B.condition_checks(R, train, "trace")
    m = R.shape[1]
    assert np.all(kappa <= bound * (1 + 1e-9)) and np.all(bound <= m * m * kappa * (1 + 1e-9))


def test_ill_conditioned_draws_go_to_scipy():
    # with a limit of 1, every draw fails the check and is refitted by SciPy
    R, s, train = awkward_problem(4)
    np.testing.assert_allclose(
        B.compiled_nnls(R, s, train, condition_limit=1.0), B.nnls_scipy_loop(R, s, train)
    )


def test_compiled_search_matches_scipy_search():
    spectra = load_arsenic()
    held_out = B.holdout_masks(len(spectra.s), draws=30)
    fast = B.search_compiled(spectra, held_out, largest=2)
    slow = B.search(spectra, held_out, largest=2, solve=B.nnls_scipy_loop)
    assert fast.keys() == slow.keys()
    for combo in slow:
        np.testing.assert_allclose(fast[combo][0], slow[combo][0], atol=1e-8)
        np.testing.assert_allclose(fast[combo][1], slow[combo][1], atol=1e-12)
