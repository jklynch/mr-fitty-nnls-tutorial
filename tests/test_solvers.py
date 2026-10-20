"""Every solver in nnls_tutorial against scipy.optimize.nnls."""

import numpy as np
import pytest
import scipy.optimize

from nnls_tutorial.data import load_arsenic
from nnls_tutorial.geometry import project_onto_cone, project_onto_span
from nnls_tutorial.solvers import (
    candidates,
    lstsq,
    nnls_enumerate,
    nnls_enumerate_many,
    support_of,
    supports,
)


def fit_error(R, s, w):
    return np.sum((R @ w - s) ** 2)


def random_problem(rng, n=30, m=None):
    """A problem where most targets want some negative weights, so NNLS has work to do."""
    m = int(rng.integers(1, 6)) if m is None else m
    R = rng.random((n, m))
    s = R @ rng.normal(size=m) + 0.05 * rng.normal(size=n)
    return R, s


def test_supports_counts_every_nonempty_subset():
    assert [len(supports(m)) for m in range(1, 6)] == [1, 3, 7, 15, 31]
    assert supports(2) == [(0,), (1,), (0, 1)]


@pytest.mark.parametrize("seed", range(200))
def test_enumerate_matches_scipy(seed):
    R, s = random_problem(np.random.default_rng(seed))
    w = nnls_enumerate(R, s)
    w_scipy, _ = scipy.optimize.nnls(R, s)
    assert np.all(w >= 0)
    np.testing.assert_allclose(w, w_scipy, atol=1e-9)
    assert abs(fit_error(R, s, w) - fit_error(R, s, w_scipy)) < 1e-12


def test_enumerate_with_duplicate_column_matches_scipy_fit():
    # the weights split between identical columns is not unique, so compare the fit
    rng = np.random.default_rng(1)
    R = rng.random((40, 3))
    R = np.column_stack([R, R[:, 0]])
    for _ in range(20):
        s = rng.random(40)
        w = nnls_enumerate(R, s)
        w_scipy, _ = scipy.optimize.nnls(R, s)
        assert abs(fit_error(R, s, w) - fit_error(R, s, w_scipy)) < 1e-12


def test_dependent_supports_are_reported_unsolved():
    R = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 0.0]])  # r_2 = 2 r_1
    by_support = {c.support: c for c in candidates(R, np.ones(3))}
    assert by_support[(0, 1)].w is None and not by_support[(0, 1)].feasible
    assert by_support[(0,)].feasible


def test_lstsq_satisfies_normal_equations():
    R, s = random_problem(np.random.default_rng(2), m=4)
    w = lstsq(R, s)
    np.testing.assert_allclose(R.T @ (s - R @ w), 0, atol=1e-12)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_enumerate_many_matches_one_at_a_time(m):
    rng = np.random.default_rng(m)
    R = rng.random((12, m))
    S = rng.normal(size=(50, 12))
    W, winner = nnls_enumerate_many(R, S)
    order = [()] + supports(m)
    for w, i, s in zip(W, winner, S, strict=True):
        np.testing.assert_allclose(w, nnls_enumerate(R, s), atol=1e-10)
        assert order[i] == support_of(w)


def test_cone_projection_is_closest_point_of_span_projection():
    # projecting s onto the cone is the same as projecting its shadow p onto the cone
    rng = np.random.default_rng(3)
    R = rng.normal(size=(3, 2))
    for _ in range(50):
        s = rng.normal(size=3)
        p, _ = project_onto_span(R, s)
        q, _ = project_onto_cone(R, s)
        q_from_p, _ = project_onto_cone(R, p)
        np.testing.assert_allclose(q, q_from_p, atol=1e-12)
        assert np.isclose(np.sum((s - q) ** 2), np.sum((s - p) ** 2) + np.sum((p - q) ** 2))


def test_bundled_best_fit_matches_mrfitty():
    # MrFitty's reported best fit of OTT3_55_spot0, from its fit summary table
    names = [
        "Arsenopyrite_Julcani_OA",
        "orpiment_all_ref_als_cal",
        "arsenate_sorbed_diop_avg_als_cal",
    ]
    spectra = load_arsenic(names)
    w = nnls_enumerate(spectra.R, spectra.s)
    np.testing.assert_allclose(w, [0.519737, 0.395842, 0.0856152], atol=1e-5)
