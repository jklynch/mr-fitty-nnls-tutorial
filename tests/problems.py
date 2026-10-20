"""Random NNLS problems shared by the test modules."""

import numpy as np


def random_problem(rng, n=30, m=None):
    """A problem where most targets want some negative weights, so NNLS has work to do."""
    m = int(rng.integers(1, 6)) if m is None else m
    R = rng.random((n, m))
    s = R @ rng.normal(size=m) + 0.05 * rng.normal(size=n)
    return R, s


def fit_error(R, s, w):
    return np.sum((R @ w - s) ** 2)
