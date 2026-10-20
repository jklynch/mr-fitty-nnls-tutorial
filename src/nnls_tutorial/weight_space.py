"""Notebook 2's figures: the NNLS problem seen in weight space.

Notebook 1 drew spectrum space, where s and the columns r_i live. Here the picture is of the
weights w = (w_1, w_2): the function f(w) = ||R w - s||^2 is a bowl over that plane, and the
constraint w >= 0 is the quadrant. The quadrant is shaded in the same color as the cone in
spectrum space, because R maps the one onto the other.

As in spectrum_space.py, every widget has a static counterpart drawn by the same function.
"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MaxNLocator

from . import plotting as P
from .geometry import two_columns, unit
from .solvers import candidates, gradient, lstsq, nnls_enumerate, objective

SLIDER_STYLE = {"description_width": "initial"}

SUPPORT_LABELS = {(): r"$\varnothing$", (0,): r"$\{1\}$", (1,): r"$\{2\}$", (0, 1): r"$\{1,2\}$"}

# The four positions of s used for Figure 1.5, one in each region, at a 60° angle between
# the columns. Reusing them lets notebook 2 point back at that figure.
REGION_EXAMPLES = {
    "inside the cone": (60, 45.0, 1.2),
    "below $r_1$": (60, -35.0, 1.2),
    "beyond $r_2$": (60, 125.0, 1.2),
    "facing away": (60, 225.0, 1.2),
}


def problem(theta, s_angle, s_length):
    """(R, s): two unit columns `theta` degrees apart, and s at the given direction and length."""
    return two_columns(theta), s_length * unit(s_angle)


def support_label(support):
    """A support as a set of column numbers, counting from 1."""
    if support in SUPPORT_LABELS:
        return SUPPORT_LABELS[support]
    return r"$\{" + ",".join(str(i + 1) for i in support) + r"\}$"


def weight_axes(ax, center, half_width, ticks=None):
    """Square axes on weight space, with the axes w_1 and w_2 drawn through the origin."""
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_aspect("equal")
    ax.axhline(0, color=P.INK_MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=P.INK_MUTED, lw=0.8, zorder=1)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.grid(False)
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:  # a few round ticks are enough to give the scale
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))


def window(points, min_half_width=0.8):
    """A square window around the given points, with some room to spare."""
    points = np.asarray(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    half_width = max(0.65 * np.max(hi - lo), min_half_width)
    return center, half_width


def draw_bowl(ax, R, s, center, half_width, n=241, n_levels=12, highlight=None):
    """Contours of f(w) = ||R w - s||^2, with the quadrant w >= 0 shaded.

    The levels are spaced as squares (f_min + j^2 * step), so the contours sit evenly apart, as
    they would for a cone rather than a bowl. `highlight`, a weight vector, also gets its own
    contour, drawn bold: for w* it's the lowest contour that touches the quadrant.
    """
    big = 10 * (abs(center).max() + half_width)
    ax.add_patch(Rectangle((0, 0), big, big, facecolor=P.REGION_FILLS[(0, 1)], zorder=0))
    g = np.linspace(-half_width, half_width, n)
    W1, W2 = np.meshgrid(center[0] + g, center[1] + g)
    pts = np.column_stack([W1.ravel(), W2.ravel()])
    F = np.sum((pts @ R.T - s) ** 2, axis=1).reshape(n, n)
    f_min = objective(R, s, lstsq(R, s))
    step = (F.max() - f_min) / n_levels**2
    levels = f_min + step * np.arange(1, n_levels + 1) ** 2
    ax.contour(W1, W2, F, levels=levels, colors=P.INK_MUTED, linewidths=0.7, zorder=2)
    if highlight is not None:
        level = objective(R, s, highlight)
        if level > f_min + 1e-9:
            ax.contour(W1, W2, F, levels=[level], colors=P.NNLS_COLOR, linewidths=1.6, zorder=3)


def _mark_weights(ax, w_hat, w_star):
    """The least-squares weights (hollow square) and the NNLS weights (red dot)."""
    ax.plot(*w_hat, **{**P.LS_STYLE, "linestyle": "none"}, ms=10, markeredgewidth=2, zorder=5)
    ax.plot(*w_star, **{**P.NNLS_STYLE, "linestyle": "none"}, ms=9, zorder=6)
    if np.allclose(w_hat, w_star):
        ax.annotate(
            r"$\hat w = w^*$", w_star, xytext=(9, 7), textcoords="offset points", fontsize=12
        )
    else:
        ax.annotate(r"$\hat w$", w_hat, xytext=(9, 7), textcoords="offset points", fontsize=12)
        ax.annotate("$w^*$", w_star, xytext=(9, 7), textcoords="offset points", fontsize=12)


def _spectrum_panel(ax, R, s, show_lines=False, cands=None):
    """Spectrum space, as in notebook 1: the cone, the columns, s and its NNLS projection q.

    With `show_lines`, the full lines through r_1 and r_2 are drawn too, and with `cands` each
    candidate's fitted point R w on them: a candidate is infeasible exactly when its point sits
    on the backward half of its line, outside the cone.
    """
    ax.add_patch(
        Polygon(P.wedge_outline(R, 6), facecolor=P.REGION_FILLS[(0, 1)], edgecolor="none", zorder=0)
    )
    for i in range(2):
        if show_lines:
            ax.axline((0, 0), R[:, i], color=P.COLUMN_COLORS[i], lw=0.8, ls="--", alpha=0.7)
        P.arrow(ax, (0, 0), R[:, i], P.COLUMN_COLORS[i], P.column_label(i))
    w_star = nnls_enumerate(R, s)
    if cands is not None:
        for cand in cands:
            if cand.support in ((), (0, 1)) or np.allclose(cand.w, w_star):
                continue  # the origin, s itself (two columns fit s exactly here), and q
            point = R @ cand.w
            _candidate_marker(ax, point, cand.feasible, winner=False)
    q = R @ w_star
    ax.plot(*zip(s, q, strict=True), color=P.NNLS_COLOR, lw=1.4, ls=":")
    ax.plot(*q, **{**P.NNLS_STYLE, "linestyle": "none"}, ms=10, zorder=5)
    ax.plot(*s, "o", color=P.INK, ms=5, zorder=6)
    ax.annotate("$s$", s, xytext=(7, 5), textcoords="offset points", fontsize=13)
    P.square_limits(ax, 2.0)
    P.coordinate_axes(ax)
    ax.set_title("Spectrum space")


def _candidate_marker(ax, point, feasible, winner):
    if winner:
        ax.plot(*point, **{**P.NNLS_STYLE, "linestyle": "none"}, ms=9, zorder=7)
    elif feasible:
        ax.plot(
            *point, "o", color=P.INK, ms=8, markerfacecolor=P.SURFACE, markeredgewidth=1.6, zorder=6
        )
    else:
        ax.plot(*point, "x", color=P.INK_SECONDARY, ms=9, markeredgewidth=2, zorder=6)


# ---------------------------------------------------------------------------------------------
# Figure 2.1: the same problem in spectrum space and in weight space


def two_spaces_figure(theta=60, s_angle=-35, s_length=1.2):
    """Spectrum space on the left, weight space on the right, for one choice of s."""
    R, s = problem(theta, s_angle, s_length)
    w_hat, w_star = lstsq(R, s), nnls_enumerate(R, s)
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(12, 5.6))
    _spectrum_panel(ax_s, R, s)
    center, half_width = window([np.zeros(2), w_hat, w_star])
    draw_bowl(ax_w, R, s, center, half_width, highlight=w_star)
    _mark_weights(ax_w, w_hat, w_star)
    weight_axes(ax_w, center, half_width)
    ax_w.set_title("Weight space")
    return fig


def two_spaces_static():
    """Figure 2.1 for s below r_1, where NNLS changes the answer, and inside the cone."""
    for key in ["below $r_1$", "inside the cone"]:
        P.show(two_spaces_figure(*REGION_EXAMPLES[key]))


# ---------------------------------------------------------------------------------------------
# Figure 2.2: the gradient field


def gradient_figure(theta=60, s_angle=-35, s_length=1.2, n_arrows=9):
    """Downhill arrows -grad f on a grid over the bowl, each scaled to the same length."""
    R, s = problem(theta, s_angle, s_length)
    w_hat = lstsq(R, s)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    center, half_width = window([np.zeros(2), w_hat], min_half_width=1.2)
    draw_bowl(ax, R, s, center, half_width)
    g = np.linspace(-0.85 * half_width, 0.85 * half_width, n_arrows)
    W1, W2 = np.meshgrid(center[0] + g, center[1] + g)
    downhill = -np.array([gradient(R, s, w) for w in np.column_stack([W1.ravel(), W2.ravel()])])
    length = np.linalg.norm(downhill, axis=1, keepdims=True)
    downhill = np.divide(downhill, length, out=np.zeros_like(downhill), where=length > 1e-12)
    ax.quiver(
        W1.ravel(),
        W2.ravel(),
        downhill[:, 0],
        downhill[:, 1],
        color=P.INK_SECONDARY,
        angles="xy",
        scale_units="xy",
        scale=1 / (0.12 * half_width),
        width=0.004,
        zorder=4,
    )
    ax.plot(*w_hat, **{**P.LS_STYLE, "linestyle": "none"}, ms=10, markeredgewidth=2, zorder=5)
    ax.annotate(r"$\hat w$", w_hat, xytext=(9, 7), textcoords="offset points", fontsize=12)
    weight_axes(ax, center, half_width)
    ax.set_title(r"Downhill directions $-\nabla f$, all drawn the same length")
    return fig


# ---------------------------------------------------------------------------------------------
# Figure 2.3: the optimality conditions at w*, in the three cases


def kkt_figure():
    """w* inside the quadrant, on an edge, and at the corner, with the downhill direction there."""
    cases = [
        ("inside the cone", r"inside: $\nabla f(w^*) = 0$"),
        ("below $r_1$", r"on an edge: $-\nabla f$ points out"),
        ("facing away", r"at the corner: $-\nabla f$ points out"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (key, title) in zip(axes, cases, strict=True):
        R, s = problem(*REGION_EXAMPLES[key])
        w_hat, w_star = lstsq(R, s), nnls_enumerate(R, s)
        center, half_width = window([np.zeros(2), w_hat, w_star], min_half_width=1.0)
        draw_bowl(ax, R, s, center, half_width, highlight=w_star)
        downhill = -gradient(R, s, w_star)
        size = np.linalg.norm(downhill)
        if size > 1e-9:
            tip = w_star + 0.45 * half_width * downhill / size
            P.arrow(ax, w_star, tip, P.NNLS_COLOR)
            ax.annotate(
                r"$-\nabla f(w^*)$",
                tip,
                xytext=(6, -14),
                textcoords="offset points",
                color=P.INK,
                fontsize=12,
            )
        _mark_weights(ax, w_hat, w_star)
        weight_axes(ax, center, half_width)
        ax.set_title(title, fontsize=12)
    return fig


# ---------------------------------------------------------------------------------------------
# Figure 2.4: every candidate, one per support


def candidates_figure(theta=60, s_angle=-35, s_length=1.2, ax=None, with_spectrum=None):
    """Weight space with the candidate for every support: crossed out if it has a negative
    weight, hollow if allowed, red for the winner. Each is labeled with its error f(w)."""
    R, s = problem(theta, s_angle, s_length)
    cands = candidates(R, s)
    w_star = nnls_enumerate(R, s)
    own_figure = ax is None
    if own_figure:
        if with_spectrum is None:
            with_spectrum = True
        if with_spectrum:
            fig, (ax_s, ax) = plt.subplots(1, 2, figsize=(12, 5.6))
            _spectrum_panel(ax_s, R, s, show_lines=True, cands=cands)
        else:
            fig, ax = plt.subplots(figsize=(6, 5.6))
    points = [np.zeros(2)] + [cand.w for cand in cands if cand.w is not None]
    center, half_width = window(points, min_half_width=1.0)
    draw_bowl(ax, R, s, center, half_width, highlight=w_star)
    for cand in cands:
        winner = np.allclose(cand.w, w_star)
        _candidate_marker(ax, cand.w, cand.feasible, winner)
        offset = (8, 7) if cand.support == (0,) else (8, -14)  # keeps {1} clear of the empty set
        ax.annotate(
            f"{support_label(cand.support)}  {cand.error:.2f}",
            cand.w,
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            color=P.INK if cand.feasible else P.INK_MUTED,
        )
    weight_axes(ax, center, half_width)
    ax.set_title(f"Candidates: the winner is {support_label(_support(w_star))}")
    if own_figure:
        return fig


def _support(w):
    return tuple(int(i) for i in np.flatnonzero(w > 1e-12))


def candidates_static():
    """Figure 2.4 in weight space only, for s in each of Figure 1.5's four regions."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0))
    for ax, (key, settings) in zip(axes, REGION_EXAMPLES.items(), strict=True):
        candidates_figure(*settings, ax=ax)
        ax.set_title(f"$s$ {key}\n{ax.get_title()}", fontsize=11)
    P.show(fig)


def candidates_widget():
    """Move s and change the angle between the columns; every candidate is recomputed."""
    return widgets.interactive(
        lambda **kw: P.show(candidates_figure(**kw)),
        theta=widgets.IntSlider(
            value=60,
            min=15,
            max=165,
            step=5,
            description="angle between $r_1$ and $r_2$ (°)",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
        s_angle=widgets.IntSlider(
            value=-35,
            min=-180,
            max=180,
            step=5,
            description="direction of $s$ (°)",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
        s_length=widgets.FloatSlider(
            value=1.2,
            min=0.2,
            max=2.0,
            step=0.1,
            description="length of $s$",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
    )


# ---------------------------------------------------------------------------------------------
# Figure 2.5: the candidates for real spectra


def real_candidates_figure(spectra, trios):
    """For each trio of references, every candidate's error on a log scale, marked as in Figure
    2.4: crossed out if it has a negative weight, hollow if allowed, red for the winner."""
    fig, axes = plt.subplots(1, len(trios), figsize=(6.6 * len(trios), 4.4), squeeze=False)
    for ax, names in zip(axes[0], trios, strict=True):
        columns = [spectra.names.index(name) for name in names]
        R, s = spectra.R[:, columns], spectra.s
        cands = candidates(R, s)
        w_star = nnls_enumerate(R, s)
        rows = np.arange(len(cands))
        for row, cand in zip(rows, cands, strict=True):
            winner = cand.feasible and np.allclose(cand.w, w_star)
            ax.hlines(row, 0.05, cand.error, color=P.GRID, lw=1, zorder=1)
            _candidate_marker(ax, (cand.error, row), cand.feasible, winner)
            note = "" if cand.feasible else "  (negative weight)"
            ax.annotate(
                f"{cand.error:.2f}{note}",
                (cand.error, row),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=P.INK if cand.feasible else P.INK_MUTED,
            )
        ax.set_xscale("log")
        ax.set_xlim(0.05, 2000)
        ax.set_yticks(rows, [support_label(cand.support) for cand in cands])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(r"error $f(w) = \|Rw - s\|^2$ (log scale)")
        labels = [spectra.labels[j] for j in columns]
        ax.set_title(
            ", ".join(f"$r_{i + 1}$ {label}" for i, label in enumerate(labels)), fontsize=10.5
        )
    return fig
