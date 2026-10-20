"""Notebook 3's figures: the walk the active-set method takes, and what it costs.

The walk is drawn in both of the spaces from notebooks 1 and 2: in weight space, w moves from
corner to edge to edge of the quadrant, and in spectrum space, R w moves along the cone toward s.
As in spectrum_space.py and weight_space.py, every widget has a static counterpart drawn by the
same function.
"""

import time

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from . import plotting as P
from . import weight_space as W
from .geometry import unit
from .solvers import active_set_steps, nnls_active_set, nnls_enumerate_normal

SLIDER_STYLE = {"description_width": "initial"}

# The example: r_1 is twice as long as r_2 and 40 degrees away from it, and s is just beyond
# r_2. r_1 lines up with s better by the dot product, because it's longer, so the walk switches
# it on first, and then has to switch it off again.
WALK_EXAMPLE = dict(r1_length=2.0, theta=40, s_angle=70, s_length=1.6)
SPECTRUM_LIMITS = (-0.6, 2.2)


def walk_problem(r1_length=2.0, theta=40, s_angle=70, s_length=1.6):
    """(R, s): r_1 at 15 degrees with the given length, a unit r_2 `theta` degrees further on."""
    R = np.column_stack([r1_length * unit(15), unit(15 + theta)])
    return R, s_length * unit(s_angle)


def describe(steps, k):
    """One line saying what step k of a recorded run did."""
    step = steps[k]
    if step.kind == "start":
        return "Start at $w = 0$, with every weight switched off"
    if step.kind == "add":
        t = step.changed[0]
        return (
            f"Switch on $w_{t + 1}$: its slope $\\lambda_{t + 1} = {step.gradient[t]:.2f}$"
            " is the most negative"
        )
    if step.kind == "fit":
        return f"The fit on {W.support_label(step.support)} is allowed: move to it"
    if step.kind == "backtrack":
        before = W.support_label(tuple(sorted(step.support + step.changed)))
        dropped = ", ".join(f"$w_{i + 1}$" for i in step.changed)
        return f"The fit on {before} isn't allowed: stop at the fence, switch off {dropped}"
    return "Stop: no switched-off weight has a negative slope"


def _path(steps, k):
    """The distinct points visited by steps 0..k, in order."""
    points = [steps[0].w]
    for step in steps[1 : k + 1]:
        if not np.allclose(step.w, points[-1]):
            points.append(step.w)
    return np.array(points)


def _draw_walk_weights(ax, R, s, steps, k, center, half_width):
    step = steps[k]
    W.draw_bowl(ax, R, s, center, half_width, highlight=step.w)
    path = _path(steps, k)
    ax.plot(*path.T, color=P.INK, lw=1.6, zorder=4)
    for j, point in enumerate(path):
        ax.annotate(
            str(j), point, xytext=(-13, 5), textcoords="offset points", fontsize=10, color=P.INK
        )
    _draw_target(ax, steps, k, lambda w: w)
    if step.kind == "add" or (k > 0 and steps[k - 1].kind == "add" and step.kind != "stop"):
        before = steps[k - 1] if step.kind != "add" else step
        downhill = -before.gradient
        tip = before.w + 0.3 * half_width * downhill / np.linalg.norm(downhill)
        P.arrow(ax, before.w, tip, P.INK_SECONDARY)
        ax.annotate(
            r"$-\nabla f$",
            tip,
            xytext=(4, 4),
            textcoords="offset points",
            color=P.INK_SECONDARY,
            fontsize=11,
        )
    done = step.kind == "stop"
    style = P.NNLS_STYLE if done else dict(color=P.INK, marker="o")
    ax.plot(*step.w, **{**style, "linestyle": "none"}, ms=9, zorder=7)
    ax.annotate(
        "$w^*$" if done else "$w$", step.w, xytext=(8, -15), textcoords="offset points", fontsize=12
    )
    W.weight_axes(ax, center, half_width)


def _draw_target(ax, steps, k, to_plot):
    """The least-squares fit z that step k aimed at, as a hollow square, with the line of aim."""
    step = steps[k]
    if step.target is None:
        return
    start, target = to_plot(steps[k - 1].w), to_plot(step.target)
    ax.plot(*zip(start, target, strict=True), color=P.LS_COLOR, lw=1.3, ls="--", zorder=3)
    ax.plot(*target, **{**P.LS_STYLE, "linestyle": "none"}, ms=10, markeredgewidth=2, zorder=5)
    ax.annotate("$z$", target, xytext=(9, 5), textcoords="offset points", fontsize=12)


def _draw_walk_spectra(ax, R, s, steps, k):
    step = steps[k]
    ax.add_patch(
        Polygon(
            P.wedge_outline(R, 10), facecolor=P.REGION_FILLS[(0, 1)], edgecolor="none", zorder=0
        )
    )
    path = _path(steps, k) @ R.T
    ax.plot(*path.T, color=P.INK, lw=1.6, zorder=2)
    # r_2's label goes to the left of its tip, clear of the walk, which runs on past it
    for i, offset in enumerate([(6, 6), (-24, -2)]):
        P.arrow(ax, (0, 0), R[:, i], P.COLUMN_COLORS[i])
        ax.annotate(
            P.column_label(i), R[:, i], xytext=offset, textcoords="offset points", fontsize=13
        )
    _draw_target(ax, steps, k, lambda w: R @ w)
    fit = R @ step.w
    ax.plot(*zip(fit, s, strict=True), color=P.INK_MUTED, lw=1.2, ls=":", zorder=3)
    done = step.kind == "stop"
    style = P.NNLS_STYLE if done else dict(color=P.INK, marker="o")
    ax.plot(*fit, **{**style, "linestyle": "none"}, ms=9, zorder=7)
    ax.annotate(
        "$q = Rw^*$" if done else "$Rw$",
        fit,
        xytext=(8, -15),
        textcoords="offset points",
        fontsize=12,
    )
    ax.plot(*s, "o", color=P.INK, ms=5, zorder=6)
    ax.annotate("$s$", s, xytext=(-14, 4), textcoords="offset points", fontsize=13)
    ax.set_xlim(*SPECTRUM_LIMITS)
    ax.set_ylim(*SPECTRUM_LIMITS)
    ax.set_aspect("equal")
    ax.axhline(0, color=P.INK_MUTED, lw=0.8, zorder=0)
    ax.axvline(0, color=P.INK_MUTED, lw=0.8, zorder=0)
    P.coordinate_axes(ax, xticks=(0, 1, 2), yticks=(0, 1, 2))


def _frame(steps):
    """A fixed window for the whole run, so the picture doesn't jump between steps."""
    points = [step.w for step in steps] + [step.target for step in steps if step.target is not None]
    return W.window(points, min_half_width=0.9)


def walk_figure(step=None, r1_length=2.0, theta=40, s_angle=70, s_length=1.6):
    """Step `step` of the active-set walk (the last step if None), in both spaces."""
    R, s = walk_problem(r1_length, theta, s_angle, s_length)
    steps = active_set_steps(R, s)
    k = len(steps) - 1 if step is None else min(step, len(steps) - 1)
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(12, 5.8))
    _draw_walk_spectra(ax_s, R, s, steps, k)
    _draw_walk_weights(ax_w, R, s, steps, k, *_frame(steps))
    ax_s.set_title("Spectrum space")
    ax_w.set_title("Weight space")
    fig.suptitle(f"Step {k} of {len(steps) - 1}. {describe(steps, k)}", fontsize=12.5)
    return fig


def walk_static():
    """The walk's steps that move w, in weight space (top) and spectrum space (bottom)."""
    R, s = walk_problem(**WALK_EXAMPLE)
    steps = active_set_steps(R, s)
    shown = [k for k, step in enumerate(steps) if step.kind in ("fit", "backtrack", "stop")]
    fig, axes = plt.subplots(2, len(shown), figsize=(4.6 * len(shown), 9.4))
    center, half_width = _frame(steps)
    for col, k in enumerate(shown):
        _draw_walk_weights(axes[0, col], R, s, steps, k, center, half_width)
        _draw_walk_spectra(axes[1, col], R, s, steps, k)
        title = describe(steps, k).replace(": ", ":\n", 1)
        axes[0, col].set_title(f"Step {k}. {title}", fontsize=10.5)
    P.show(fig)


def walk_widget():
    """Step through the walk, and move s to see other walks."""
    return widgets.interactive(
        lambda **kw: P.show(walk_figure(**kw)),
        step=widgets.IntSlider(
            value=0,
            min=0,
            max=8,
            description="step",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
        s_angle=widgets.IntSlider(
            value=WALK_EXAMPLE["s_angle"],
            min=-180,
            max=180,
            step=5,
            description="direction of $s$ (°)",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
        r1_length=widgets.FloatSlider(
            value=WALK_EXAMPLE["r1_length"],
            min=0.5,
            max=3.0,
            step=0.1,
            description="length of $r_1$",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
    )


# ---------------------------------------------------------------------------------------------
# The walk on real spectra


def support_chart_figure(spectra, columns=None):
    """Which weights are on after each move of the walk, and the error so far, for real spectra.

    Each column of dots is the weights after a fit or a backtrack. A dot's area shows its
    weight, and a cross marks a weight that a backtrack switched off.
    """
    columns = list(range(len(spectra.names))) if columns is None else columns
    R, s = spectra.R[:, columns], spectra.s
    steps = active_set_steps(R, s)
    moves = [step for step in steps if step.kind in ("start", "fit", "backtrack")]
    labels = [spectra.labels[j] for j in columns]
    fig, (ax_f, ax) = plt.subplots(
        2, 1, figsize=(9.5, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.6])
    )
    x = np.arange(len(moves))
    ax_f.plot(x, [step.error for step in moves], color=P.INK, marker="o", ms=5, lw=1.5)
    ax_f.set_yscale("log")
    ax_f.set_ylabel("error $f(w)$")
    biggest = max(step.w.max() for step in moves)
    for col, step in enumerate(moves):
        for i in step.support:
            ax.scatter(col, i, s=900 * step.w[i] / biggest + 12, color=P.NNLS_COLOR, zorder=3)
        for i in step.changed:
            ax.plot(col, i, "x", color=P.INK, ms=11, markeredgewidth=2.2, zorder=4)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xticks(x, [step.kind for step in moves], rotation=40, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(moves) - 0.4)
    return fig


# ---------------------------------------------------------------------------------------------
# What each method costs


def _seconds_per_call(solve, R, s, budget=0.005):
    """The best of three timings of solve(R, s), each over enough calls to fill `budget`."""
    start = time.perf_counter()
    solve(R, s)
    calls = max(1, int(budget / max(time.perf_counter() - start, 1e-7)))
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(calls):
            solve(R, s)
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def measure_costs(spectra, subsets_per_size=4, seed=0):
    """For each number of references m, some random m-subsets of the pool, fit both ways.

    Returns one row per (subset, method): the number of linear systems solved, and the
    seconds per fit with this package's plain-Python solvers.
    """
    rng = np.random.default_rng(seed)
    pool = len(spectra.names)
    rows = []
    for m in range(1, pool + 1):
        for _ in range(subsets_per_size if m < pool else 1):
            columns = sorted(rng.choice(pool, size=m, replace=False))
            R, s = spectra.R[:, columns], spectra.s
            steps = active_set_steps(R, s)
            rows.append(
                dict(
                    m=m,
                    method="active set",
                    systems=sum(step.kind in ("fit", "backtrack") for step in steps),
                    seconds=_seconds_per_call(nnls_active_set, R, s),
                )
            )
            rows.append(
                dict(
                    m=m,
                    method="every support",
                    systems=2**m - 1,
                    seconds=_seconds_per_call(nnls_enumerate_normal, R, s),
                )
            )
    return pd.DataFrame(rows)


def cost_figure(costs):
    """Systems solved (left) and time per fit (right) against m, for both methods."""
    fig, (ax_n, ax_t) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"every support": P.LS_COLOR, "active set": P.NNLS_COLOR}
    for method, rows in costs.groupby("method"):
        typical = rows.groupby("m").median(numeric_only=True)
        for ax, column in [(ax_n, "systems"), (ax_t, "seconds")]:
            ax.scatter(rows["m"], rows[column], color=colors[method], s=14, alpha=0.35, lw=0)
            ax.plot(typical.index, typical[column], color=colors[method], marker="o", label=method)
    for ax in (ax_n, ax_t):
        ax.set_yscale("log")
        ax.set_xlabel("references $m$")
        ax.set_xticks(range(1, costs["m"].max() + 1))
        ax.legend()
    ax_n.set_ylabel("linear systems solved")
    ax_t.set_ylabel("seconds per fit")
    ax_n.set_title("Work: the same on any computer")
    ax_t.set_title("Time: on the computer that ran this notebook")
    return fig
