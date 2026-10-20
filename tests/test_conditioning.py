"""The conditioning facts notebook 4 proves, checked numerically."""

import numpy as np

from nnls_tutorial.conditioning import (
    cond,
    cond_columns,
    digits_lost_experiment,
    support_conds,
    trace_bound,
    unit_pair,
)


def test_gram_matrix_squares_the_condition_number():
    # Theorem 18
    for seed in range(20):
        R = np.random.default_rng(seed).random((30, 4))
        assert np.isclose(cond(R.T @ R), cond_columns(R) ** 2, rtol=1e-8), f"seed {seed}"


def test_trace_bound_brackets_the_condition_number():
    # Theorem 22
    for seed in range(20):
        R = np.random.default_rng(seed).random((30, 5))
        G = R.T @ R
        bound = trace_bound(G)
        message = f"seed {seed}"
        assert np.isclose(bound, np.trace(G) * np.trace(np.linalg.inv(G)), rtol=1e-8), message
        assert cond(G) <= bound * (1 + 1e-10), message
        assert bound <= 25 * cond(G) * (1 + 1e-10), message


def test_no_support_is_worse_conditioned_than_the_whole():
    # Theorem 21
    R = np.random.default_rng(3).random((30, 6))
    G = R.T @ R
    assert max(support_conds(G).values()) <= cond(G) * (1 + 1e-9)


def test_unit_pair_has_the_cotangent_condition_number():
    # Exercise 1 of notebook 4: kappa(R) = cot(theta / 2)
    for theta in [90, 30, 5, 0.5]:
        assert np.isclose(cond_columns(unit_pair(theta)), 1 / np.tan(np.radians(theta) / 2))


def test_normal_equations_lose_about_twice_the_digits_of_qr():
    normal, qr = digits_lost_experiment(unit_pair(0.01))
    assert 6 < normal < 9.5 and qr < 4.5
