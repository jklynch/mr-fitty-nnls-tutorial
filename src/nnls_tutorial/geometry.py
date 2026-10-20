"""Projections onto a span and onto a cone, and the small examples the pictures use."""

import numpy as np

from .solvers import lstsq, nnls_enumerate


def project_onto_span(R, s):
    """(p, w): the closest point p = R w to s in the column space of R, and its weights."""
    w = lstsq(R, s)
    return R @ w, w


def project_onto_cone(R, s):
    """(q, w): the closest point q = R w to s with w >= 0, and its weights."""
    w = nnls_enumerate(R, s)
    return R @ w, w


def unit(angle_degrees):
    """The unit vector in the plane at the given angle from the positive x-axis."""
    t = np.radians(angle_degrees)
    return np.array([np.cos(t), np.sin(t)])


def two_columns(theta, first=15.0):
    """A 2 x 2 matrix whose unit columns sit at `first` degrees and `first + theta` degrees."""
    return np.column_stack([unit(first), unit(first + theta)])


def mixture_coordinates(w):
    """Weights scaled to sum to 1: where R w points, as a mixture of the columns.

    For three columns these are barycentric coordinates in the triangle whose corners are the
    columns themselves. A negative weight puts the point outside the triangle.
    """
    w = np.asarray(w, float)
    return w / w.sum()


# Corners of an equilateral triangle, one per column, for drawing mixtures of three columns.
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
