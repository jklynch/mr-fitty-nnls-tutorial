"""Notebook 4's figures: how much a solve can lose to rounding, and the Cholesky factorization.

As in the other figure modules, every widget has a static counterpart drawn by the same function.
"""

import functools

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import plotting as P
from .conditioning import (
    cond,
    cond_columns,
    digits_lost_experiment,
    support_conds,
    trace_bound,
    unit_pair,
)
from .geometry import two_columns

SLIDER_STYLE = {"description_width": "initial"}
NORMAL_STYLE = dict(color=P.INK, marker="o", ms=6, lw=1.6)
QR_STYLE = dict(color=P.INK_SECONDARY, marker="s", ms=6, lw=1.6, ls="--", markerfacecolor=P.SURFACE)

# The two most nearly identical references in the bundled pool.
TWINS = ["As2O3_ref_avg_als_cal", "sodium_arsenite_13BM_ref_als_cal"]


# ---------------------------------------------------------------------------------------------
# Figure 4.1: stretching, and the digits it costs


def _ellipse_panel(ax, theta):
    """Two unit columns theta degrees apart, and where R sends the unit circle of weights."""
    R = two_columns(theta)
    t = np.linspace(0, 2 * np.pi, 361)
    image = R @ np.vstack([np.cos(t), np.sin(t)])
    ax.fill(*image, color=P.REGION_FILLS[(0, 1)], zorder=0)
    ax.plot(*image, color=P.INK_MUTED, lw=1, zorder=1)
    U, sigma, _ = np.linalg.svd(R)
    for k in range(2):  # the ellipse's long and short half-axes
        ax.plot(*zip((0, 0), sigma[k] * U[:, k], strict=True), color=P.INK, lw=1.8, zorder=3)
    # r_1's label goes below its tip and r_2's above, so they stay apart when the angle is small
    for i, offset in enumerate([(4, -16), (-8, 8)]):
        P.arrow(ax, (0, 0), R[:, i], P.COLUMN_COLORS[i])
        ax.annotate(
            P.column_label(i), R[:, i], xytext=offset, textcoords="offset points", fontsize=13
        )
    P.square_limits(ax, 1.6)
    P.coordinate_axes(ax)
    ax.set_title(
        f"{theta:g}° apart: $\\sigma_{{\\max}} = {sigma[0]:.2g}$, "
        f"$\\sigma_{{\\min}} = {_latex_number(sigma[1])}$\n"
        f"$\\kappa(R) = {_latex_number(cond_columns(R))}$",
        fontsize=11.5,
    )


def _latex_number(x):
    """x to three significant figures, as 1.23 \\times 10^{4} once it's large or small."""
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.3g}"
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"{float(mantissa):g} \\times 10^{{{int(exponent)}}}"


@functools.cache
def digits_lost_curve():
    """(kappa(R), digits lost by the normal equations, digits lost by QR) over many angles."""
    angles = np.logspace(np.log10(90), -7, 29)
    rows = [(cond_columns(unit_pair(a)), *digits_lost_experiment(unit_pair(a))) for a in angles]
    return tuple(np.array(column) for column in zip(*rows, strict=True))


def _digits_panel(ax, theta=None):
    kappa, normal, qr = digits_lost_curve()
    ax.plot(kappa, np.minimum(2 * np.log10(kappa), 16), color=P.GRID, lw=6, zorder=0)
    ax.plot(kappa, np.log10(kappa), color=P.GRID, lw=6, zorder=0)
    ax.annotate(
        r"$2\log_{10}\kappa(R)$",
        (3e5, 2 * np.log10(3e5)),
        xytext=(-60, 10),
        textcoords="offset points",
        color=P.INK_MUTED,
        fontsize=10,
    )
    ax.annotate(
        r"$\log_{10}\kappa(R)$",
        (3e6, np.log10(3e6)),
        xytext=(-30, 12),
        textcoords="offset points",
        color=P.INK_MUTED,
        fontsize=10,
    )
    ax.plot(kappa, normal, **NORMAL_STYLE, label="normal equations, by Cholesky")
    ax.plot(kappa, qr, **QR_STYLE, label="QR")
    ax.axhline(16, color=P.INK_MUTED, lw=0.8, ls=":")
    ax.annotate(
        "all 16 digits gone",
        (1.5, 16),
        xytext=(0, -14),
        textcoords="offset points",
        color=P.INK_MUTED,
        fontsize=10,
    )
    if theta is not None:
        here = cond_columns(unit_pair(theta))
        ax.axvline(here, color=P.NNLS_COLOR, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlim(1, kappa.max() * 1.5)
    ax.set_ylim(-0.5, 17)
    ax.set_xlabel(r"$\kappa(R)$")
    ax.set_ylabel("digits lost")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.86))
    ax.set_title("Fitting $s = Rw$ exactly, two unit columns in $\\mathbb{R}^{50}$", fontsize=11.5)


def stretch_figure(theta=5.0):
    """The stretching picture for one angle, and the digits-lost curve with that angle marked."""
    fig, (ax_e, ax_d) = plt.subplots(1, 2, figsize=(12.5, 5.4), width_ratios=[1, 1.35])
    _ellipse_panel(ax_e, theta)
    _digits_panel(ax_d, theta)
    return fig


def stretch_static():
    """The stretching picture at three angles, then the digits-lost curve."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    for ax, theta in zip(axes, [90, 30, 5], strict=True):
        _ellipse_panel(ax, theta)
    P.show(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _digits_panel(ax)
    P.show(fig)


def stretch_widget():
    """Close the angle between the columns and watch the ellipse flatten and the digits go."""
    return widgets.interactive(
        lambda theta: P.show(stretch_figure(theta)),
        theta=widgets.FloatLogSlider(
            value=5,
            base=10,
            min=-7,
            max=np.log10(90),
            step=0.1,
            description="angle between $r_1$ and $r_2$ (°)",
            continuous_update=False,
            style=SLIDER_STYLE,
            readout_format=".2g",
        ),
    )


# ---------------------------------------------------------------------------------------------
# Figure 4.2: the twins


def twins_figure(spectra):
    """The pool's two most nearly identical references, and how they differ."""
    columns = [spectra.names.index(name) for name in TWINS]
    R = spectra.R[:, columns]
    labels = [spectra.labels[j] for j in columns]
    fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for k, label in enumerate(labels):
        ax_r.plot(spectra.energies, R[:, k], color=P.COLUMN_COLORS[k], lw=1.6, label=label)
    ax_r.set_xlabel("energy (eV)")
    ax_r.set_ylabel("normalized absorption")
    ax_r.legend()
    cosine = R[:, 0] @ R[:, 1] / np.linalg.norm(R[:, 0]) / np.linalg.norm(R[:, 1])
    ax_r.set_title(
        f"{np.degrees(np.arccos(cosine)):.2f}° apart: $\\kappa(R) = {cond_columns(R):.0f}$",
        fontsize=11.5,
    )
    unit = R / np.linalg.norm(R, axis=0)
    ax_d.plot(spectra.energies, unit[:, 0] - unit[:, 1], color=P.INK, lw=1.4)
    ax_d.axhline(0, color=P.INK_MUTED, lw=0.8)
    ax_d.set_xlabel("energy (eV)")
    ax_d.set_ylabel("difference")
    ax_d.set_title("The difference, after scaling both to length 1", fontsize=11.5)
    return fig


# ---------------------------------------------------------------------------------------------
# Figure 4.3: the order Cholesky fills in L


def cholesky_order(m):
    """The entries (i, j) of L in the order they're computed: column by column, top down."""
    return [(i, j) for j in range(m) for i in range(j, m)]


def _cholesky_panel(ax, m, step, show_numbers=True):
    """L as a grid, with entries done (gray), the current entry (red) and what it reads."""
    order = cholesky_order(m)
    i, j = order[step - 1]
    reads = {(i, k) for k in range(j)} | {(j, k) for k in range(j)}
    if i != j:
        reads.add((j, j))
    for n, (a, b) in enumerate(order, start=1):
        if (a, b) == (i, j):
            face, edge = P.NNLS_COLOR, P.NNLS_COLOR
        elif (a, b) in reads:
            face, edge = P.REGION_FILLS[(0,)], P.COLUMN_COLORS[0]
        elif n < step:
            face, edge = P.GRID, P.INK_MUTED
        else:
            face, edge = P.SURFACE, P.GRID
        ax.add_patch(Rectangle((b, a), 1, 1, facecolor=face, edgecolor=edge, lw=1.4))
        if show_numbers:
            color = P.SURFACE if (a, b) == (i, j) else (P.INK if n <= step else P.INK_MUTED)
            ax.text(b + 0.5, a + 0.5, str(n), ha="center", va="center", fontsize=11, color=color)
    for a in range(m):
        for b in range(a + 1, m):  # the upper triangle of L is zero
            ax.text(b + 0.5, a + 0.5, "0", ha="center", va="center", fontsize=10, color=P.GRID)
    ax.set_xlim(0, m)
    ax.set_ylim(m, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(m) + 0.5, [f"{b + 1}" for b in range(m)])
    ax.set_yticks(np.arange(m) + 0.5, [f"{a + 1}" for a in range(m)])
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    what = "diagonal" if i == j else "below the diagonal"
    ax.set_title(f"Step {step}: $L_{{{i + 1}{j + 1}}}$, {what}", fontsize=11.5)


def cholesky_figure(step=5, m=5):
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    _cholesky_panel(ax, m, step)
    return fig


def cholesky_static(m=5):
    """Three moments in factoring a 5 x 5 matrix: a diagonal entry, one below it, and the last."""
    steps = [6, 8, len(cholesky_order(m))]
    fig, axes = plt.subplots(1, len(steps), figsize=(14, 4.9))
    for ax, step in zip(axes, steps, strict=True):
        _cholesky_panel(ax, m, step)
    P.show(fig)


def cholesky_widget(m=5):
    return widgets.interactive(
        lambda step: P.show(cholesky_figure(step, m)),
        step=widgets.IntSlider(
            value=1,
            min=1,
            max=len(cholesky_order(m)),
            description="step",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
    )


# ---------------------------------------------------------------------------------------------
# Figure 4.4: one check for every support


def support_conditioning_figure(spectra):
    """kappa(G_PP) for every support of the ten references, against the support's size."""
    G = spectra.R.T @ spectra.R
    twins = {spectra.names.index(name) for name in TWINS}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for support, kappa in support_conds(G).items():
        both = twins <= set(support)
        x = len(support) + rng.uniform(-0.28, 0.28)
        ax.plot(
            x,
            kappa,
            "o",
            ms=3.5,
            alpha=0.75,
            color=P.NNLS_COLOR if both else P.INK_MUTED,
            zorder=3 if both else 2,
        )
    full, bound = cond(G), trace_bound(G)
    ax.axhline(full, color=P.INK, lw=1.4, ls="--")
    ax.axhline(bound, color=P.LS_COLOR, lw=1.4, ls=":")
    ax.annotate(
        f"$\\kappa(G) = {_latex_number(full)}$, all ten references",
        (0.6, full),
        xytext=(0, -15),
        textcoords="offset points",
        fontsize=10.5,
    )
    ax.annotate(
        f"trace bound ${_latex_number(bound)}$",
        (0.6, bound),
        xytext=(0, 5),
        textcoords="offset points",
        color=P.LS_COLOR,
        fontsize=10.5,
    )
    ax.plot([], [], "o", color=P.NNLS_COLOR, label="contains both twins")
    ax.plot([], [], "o", color=P.INK_MUTED, label="doesn't")
    ax.legend(loc="lower right")
    ax.set_yscale("log")
    ax.set_xlim(0.5, len(G) + 0.5)
    ax.set_xticks(range(1, len(G) + 1))
    ax.set_xlabel("references switched on, $|P|$")
    ax.set_ylabel(r"$\kappa(G_{PP})$")
    ax.grid(axis="x", visible=False)
    return fig
