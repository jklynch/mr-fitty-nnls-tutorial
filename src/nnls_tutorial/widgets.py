"""The tutorial's figures, each as a static version and an interactive widget.

Every widget `*_widget()` has a static counterpart `*_static()` (snapshots at representative
settings), so a notebook saved with its outputs reads fully without a running kernel. Both
draw with the same `*_figure()` function.

Notation: R has the reference columns r_1, ..., r_m, s is the measured spectrum, w the weights.
"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from . import plotting as P
from .geometry import (
    TRIANGLE,
    mixture_coordinates,
    project_onto_cone,
    project_onto_span,
    two_columns,
    unit,
)
from .solvers import lstsq, nnls_enumerate, nnls_enumerate_many, supports

SLIDER_STYLE = {"description_width": "initial"}

# ---------------------------------------------------------------------------------------------
# Notebook 1: the real problem


def _weights_bars(ax, labels, w_ls, w_nnls):
    """Least-squares and NNLS weights side by side, one row per reference."""
    rows = np.arange(len(labels))
    h = 0.36
    ax.barh(rows - h / 2, w_ls, height=h, color=P.LS_COLOR, label="least squares")
    ax.barh(rows + h / 2, w_nnls, height=h, color=P.NNLS_COLOR, label="NNLS")
    for row, value in zip(rows - h / 2, w_ls, strict=True):
        ax.annotate(
            f"{value:+.2f}",
            (value, row),
            xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            color=P.INK_SECONDARY,
            fontsize=9,
        )
    for row, value in zip(rows + h / 2, w_nnls, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (value, row),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=P.INK_SECONDARY,
            fontsize=9,
        )
    ax.axvline(0, color=P.INK, lw=1)
    ax.set_yticks(rows, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lo = min(0.0, w_ls.min(), w_nnls.min())
    hi = max(w_ls.max(), w_nnls.max())
    pad = 0.25 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("weight (fraction of the mixture)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)


def teaser_figure(spectra, names):
    """The references, the unknown, and the weights least squares and NNLS give it."""
    columns = [spectra.names.index(name) for name in names]
    R, s = spectra.R[:, columns], spectra.s
    labels = [spectra.labels[i] for i in columns]
    w_ls, w_nnls = lstsq(R, s), nnls_enumerate(R, s)

    fig, (ax_spec, ax_w) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[1.6, 1])
    for i, label in enumerate(labels):
        ax_spec.plot(
            spectra.energies,
            R[:, i],
            color=P.COLUMN_COLORS[i],
            lw=1.6,
            label=f"{P.column_label(i)}  {label}",
        )
    ax_spec.plot(
        spectra.energies,
        s,
        color=P.INK,
        lw=1.2,
        marker=".",
        ms=3,
        label=f"$s$  {spectra.unknown} (measured)",
    )
    ax_spec.set_xlabel("energy (eV)")
    ax_spec.set_ylabel("normalized absorption")
    ax_spec.set_title("Three reference spectra and one measured spectrum")
    ax_spec.legend(loc="lower right")
    _weights_bars(ax_w, labels, w_ls, w_nnls)
    ax_w.set_title("How much of each reference?")
    return fig


def mixtures_figure(spectra, names):
    """A fit with three references: the fits, the weights, and where they sit in the triangle."""
    columns = [spectra.names.index(name) for name in names]
    R, s = spectra.R[:, columns], spectra.s
    labels = [spectra.labels[i] for i in columns]
    w_ls, w_nnls = lstsq(R, s), nnls_enumerate(R, s)

    fig, (ax_fit, ax_w, ax_tri) = plt.subplots(1, 3, figsize=(15, 4.4), width_ratios=[1.4, 1, 1])
    ax_fit.plot(spectra.energies, s, color=P.INK, lw=1, marker=".", ms=3, label="measured $s$")
    ax_fit.plot(
        spectra.energies,
        R @ w_ls,
        color=P.LS_COLOR,
        ls="--",
        label=f"least squares, $\\|e\\|$ = {np.linalg.norm(R @ w_ls - s):.3f}",
    )
    ax_fit.plot(
        spectra.energies,
        R @ w_nnls,
        color=P.NNLS_COLOR,
        label=f"NNLS, $\\|e\\|$ = {np.linalg.norm(R @ w_nnls - s):.3f}",
    )
    ax_fit.set_xlabel("energy (eV)")
    ax_fit.set_ylabel("normalized absorption")
    ax_fit.set_title("The two fits")
    ax_fit.legend(loc="lower right")

    _weights_bars(ax_w, labels, w_ls, w_nnls)
    ax_w.set_title("The weights")

    _mixture_triangle(ax_tri, labels, w_ls, w_nnls)
    return fig


def _mixture_triangle(ax, labels, w_ls, w_nnls):
    """Mixtures of three columns as points of a triangle; a negative weight lands outside it."""
    ax.add_patch(
        Polygon(
            TRIANGLE,
            closed=True,
            facecolor=P.REGION_FILLS[(0, 1)],
            edgecolor=P.INK_MUTED,
            lw=1,
            zorder=1,
        )
    )
    offsets = [(-10, -16), (10, -16), (0, 10)]
    aligns = ["right", "left", "center"]
    for i, (corner, label) in enumerate(zip(TRIANGLE, labels, strict=True)):
        ax.scatter(*corner, s=90, color=P.COLUMN_COLORS[i], zorder=3)
        ax.annotate(
            f"{P.column_label(i)} {label}",
            corner,
            xytext=offsets[i],
            textcoords="offset points",
            ha=aligns[i],
            color=P.INK,
            fontsize=10,
        )
    points = [corner for corner in TRIANGLE]
    for w, style, name in [(w_ls, P.LS_STYLE, "least squares"), (w_nnls, P.NNLS_STYLE, "NNLS")]:
        if w.sum() <= 0:
            continue  # no mixture to draw: the weights don't add up to anything positive
        point = mixture_coordinates(w) @ TRIANGLE
        points.append(point)
        ax.plot(
            *point,
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            markerfacecolor=style.get("markerfacecolor", style["color"]),
            ms=10,
            markeredgewidth=2,
            zorder=4,
            label=f"{name}: weights sum to {w.sum():.2f}",
        )
    points = np.array(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = 0.2 * max(hi - lo)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Each fit as a mixture of the three")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02))


def mixtures_static(spectra, presets):
    """mixtures_figure for each preset choice of three references."""
    for names in presets:
        P.show(mixtures_figure(spectra, names))


def mixtures_widget(spectra):
    """Pick any three of the bundled references and see both fits."""
    options = list(zip(spectra.labels, spectra.names, strict=True))
    defaults = ["Arsenopyrite_Julcani_OA", "As_pyrite_A_Foster_sln", "Lollingite_Lolling_avg_OA"]

    def draw(first, second, third):
        names = [first, second, third]
        if len(set(names)) < 3:
            print("Pick three different references.")
            return
        P.show(mixtures_figure(spectra, names))

    dropdowns = {
        key: widgets.Dropdown(
            options=options, value=default, description=description, style=SLIDER_STYLE
        )
        for key, default, description in zip(
            ["first", "second", "third"], defaults, ["$r_1$", "$r_2$", "$r_3$"], strict=True
        )
    }
    return widgets.interactive(draw, **dropdowns)


# ---------------------------------------------------------------------------------------------
# Notebook 1: projection in three dimensions

PLANE_COLUMNS = np.column_stack([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])  # 60° apart


def _wedge_outline(R2, radius, n=60):
    """Points outlining the sector between two unit columns in the plane, out to `radius`."""
    start = np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))
    stop = np.degrees(np.arctan2(R2[1, 1], R2[0, 1]))
    arc = [radius * unit(t) for t in np.linspace(start, stop, n)]
    return np.array([[0.0, 0.0]] + arc)


def projection3d_figure(s1=0.9, s2=0.6, s3=0.9, elev=20, azim=-62, show_cone=True):
    """s in R^3, the plane spanned by two columns, and the closest points on plane and wedge."""
    R, s = PLANE_COLUMNS, np.array([s1, s2, s3])
    p, w_ls = project_onto_span(R, s)
    q, w_nnls = project_onto_cone(R, s)

    fig = plt.figure(figsize=(12, 5.2) if show_cone else (6.5, 5.2))
    ax = fig.add_subplot(1, 2 if show_cone else 1, 1, projection="3d")
    ax.set_title(
        f"least squares: $w$ = ({w_ls[0]:.2f}, {w_ls[1]:.2f})"
        + (f"\nNNLS: $w^*$ = ({w_nnls[0]:.2f}, {w_nnls[1]:.2f})" if show_cone else "")
    )
    grid = np.linspace(-1.4, 1.8, 2)
    X, Y = np.meshgrid(grid, grid)
    ax.plot_surface(X, Y, np.zeros_like(X), color=P.INK_MUTED, alpha=0.12, linewidth=0)
    if show_cone:
        wedge = _wedge_outline(R[:2], 1.8)
        ax.add_collection3d(
            _poly3d(np.column_stack([wedge, np.zeros(len(wedge))]), P.REGION_FILLS[(0, 1)])
        )
    origin = np.zeros(3)
    for i in range(2):
        P.arrow3d(ax, origin, R[:, i], P.COLUMN_COLORS[i], P.column_label(i))
    P.arrow3d(ax, origin, s, P.INK, "$s$")
    # snap=False: pixel snapping would shift this vertical line off the dots it joins
    (drop,) = ax.plot(*zip(s, p, strict=True), color=P.LS_COLOR, ls="--", lw=1.6, snap=False)
    if show_cone:
        # a hollow square, so p stays distinct from the NNLS dot q, even where they coincide
        ax.scatter(
            *p,
            color=P.LS_COLOR,
            marker="s",
            s=50,
            facecolors="none",
            linewidths=2,
            depthshade=False,
        )
    else:
        ax.scatter(*p, color=P.LS_COLOR, s=28, depthshade=False)  # alone: a dot like the others
    same = show_cone and np.allclose(p, q)  # s is over the wedge: one point, one label
    p_label = "$p = q$" if same else "$p = Rw$"
    ax.text(*_beyond(p), p_label, color=P.INK, fontsize=12, ha="center", va="center")
    if show_cone:
        ax.plot(*zip(s, q, strict=True), color=P.NNLS_COLOR, lw=1.6)
        ax.plot(*zip(p, q, strict=True), color=P.INK_MUTED, ls=":", lw=1.4)
        ax.scatter(*q, color=P.NNLS_COLOR, s=50, depthshade=False)
        if not same:
            ax.text(*_beyond(q), "$q = Rw^*$", color=P.INK, fontsize=12, ha="center", va="center")
    ax.set_xlim(-1.4, 1.8)
    ax.set_ylim(-1.4, 1.8)
    ax.set_zlim(-0.2, 1.6)
    ax.set_box_aspect((3.2, 3.2, 1.8))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")

    if show_cone:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.add_patch(
            Polygon(
                _wedge_outline(R[:2], 6),
                facecolor=P.REGION_FILLS[(0, 1)],
                edgecolor="none",
                zorder=0,
            )
        )
        for i in range(2):
            P.arrow(ax2, (0, 0), R[:2, i], P.COLUMN_COLORS[i], P.column_label(i))
        ax2.plot(*zip(p[:2], q[:2], strict=True), color=P.INK_MUTED, ls=":", lw=1.4)
        ax2.plot(*p[:2], **{**P.LS_STYLE, "linestyle": "none"}, ms=9, markeredgewidth=2)
        ax2.plot(*q[:2], **{**P.NNLS_STYLE, "linestyle": "none"}, ms=8)
        if same:
            ax2.annotate("$p = q$", p[:2], xytext=(8, 6), textcoords="offset points", fontsize=12)
        else:
            ax2.annotate("$p$", p[:2], xytext=(8, -12), textcoords="offset points", fontsize=12)
            ax2.annotate("$q$", q[:2], xytext=(8, 6), textcoords="offset points", fontsize=12)
        P.square_limits(ax2, 1.8)
        d_sp, d_pq, d_sq = (np.sum((u - v) ** 2) for u, v in [(s, p), (p, q), (s, q)])
        ax2.set_title("The plane seen from above")
        ax2.text(
            0.02,
            0.02,
            f"$\\|s-p\\|^2 + \\|p-q\\|^2$ = {d_sp:.3f} + {d_pq:.3f}\n$\\|s-q\\|^2$ = {d_sq:.3f}",
            transform=ax2.transAxes,
            color=P.INK_SECONDARY,
            fontsize=10,
            bbox=dict(facecolor=P.SURFACE, edgecolor=P.GRID),
        )
    _fit_dashes(ax, s, p, drop)
    return fig


def _fit_dashes(ax, start, end, line, dash=3.2, gap=1.6):
    """Stretch a dashed line's pattern so it begins and ends with a dash, touching both ends.

    A dash pattern otherwise stops wherever it happens to run out, often in a gap, leaving the
    line short of its end point. The pattern is in units of the line width, and the line's
    length on screen depends on the view, so it's measured after the figure is laid out.
    """
    from mpl_toolkits.mplot3d import proj3d

    fig = ax.figure
    fig.canvas.draw()
    ends = [
        ax.transData.transform(proj3d.proj_transform(*pt, ax.get_proj())[:2]) for pt in (start, end)
    ]
    length = np.hypot(*(ends[1] - ends[0])) * 72 / fig.dpi / line.get_linewidth()
    # n dashes and n - 1 gaps, scaled together to fill the length exactly
    n = max(1, round((length + gap) / (dash + gap)))
    scale = length / (n * dash + (n - 1) * gap)
    line.set_dashes([dash * scale, gap * scale] if n > 1 else [length, 0.0])


def _beyond(point, gap=0.4):
    """Where to label a point in the plane: just past it, on the side away from the origin.

    The column labels sit at the column tips, so pushing a label outward along the point's own
    direction keeps it clear of them, wherever the sliders put the point.
    """
    radius = np.linalg.norm(point[:2])
    direction = point[:2] / radius if radius > 1e-9 else np.array([-0.7, -0.7])
    return np.array([*(point[:2] + gap * direction), point[2]])


def _poly3d(vertices, color):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return Poly3DCollection([vertices], facecolors=color, edgecolors="none", alpha=0.5)


def projection3d_static():
    """Two positions of s: one above the wedge, one whose shadow falls outside it."""
    for s in [(0.9, 0.6, 0.9), (1.4, -0.7, 0.9)]:
        P.show(projection3d_figure(*s))


def projection3d_widget():
    """Move s around R^3 and turn the view."""

    def slider(value, lo, hi, description):
        return widgets.FloatSlider(
            value=value,
            min=lo,
            max=hi,
            step=0.05,
            description=description,
            continuous_update=False,
            style=SLIDER_STYLE,
        )

    return widgets.interactive(
        lambda **kw: P.show(projection3d_figure(**kw)),
        s1=slider(1.4, -1.4, 1.8, "$s_1$"),
        s2=slider(-0.7, -1.4, 1.8, "$s_2$"),
        s3=slider(0.9, -0.2, 1.6, "$s_3$"),
        elev=widgets.IntSlider(
            value=20,
            min=-10,
            max=90,
            description="elevation",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
        azim=widgets.IntSlider(
            value=-62,
            min=-180,
            max=180,
            description="azimuth",
            continuous_update=False,
            style=SLIDER_STYLE,
        ),
        show_cone=widgets.fixed(True),
    )


# ---------------------------------------------------------------------------------------------
# Notebook 1: the four regions of the plane

REGION_NAMES = {
    (): "$w^* = 0$",
    (0,): "only $r_1$ on",
    (1,): "only $r_2$ on",
    (0, 1): "both on",
}


def regions_figure(theta=60, s_angle=-35, s_length=1.2, ax=None, radius=2.2, n=401):
    """Every point of the plane, colored by which columns its NNLS fit switches on."""
    R = two_columns(theta)
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(grid, grid)
    points = np.column_stack([X.ravel(), Y.ravel()])
    _, winner = nnls_enumerate_many(R, points)
    order = [()] + supports(2)
    ax.imshow(
        winner.reshape(n, n),
        origin="lower",
        extent=(-radius, radius, -radius, radius),
        cmap=ListedColormap([P.REGION_FILLS[support] for support in order]),
        vmin=-0.5,
        vmax=len(order) - 0.5,
        interpolation="nearest",
        zorder=0,
    )
    for i, support in enumerate(order):
        where = winner == i
        if where.sum() > 0.01 * n * n:
            # out along the region's middle direction, clear of s and the arrows
            direction = points[where].mean(axis=0)
            cx, cy = 0.7 * radius * direction / np.abs(direction).max()
            ax.text(
                cx,
                cy,
                REGION_NAMES[support],
                ha="center",
                va="center",
                color=P.INK_SECONDARY,
                fontsize=11,
            )
    for i in range(2):
        ax.plot(
            [0, 2 * radius * R[0, i]],
            [0, 2 * radius * R[1, i]],
            color=P.COLUMN_COLORS[i],
            lw=1,
            alpha=0.6,
        )
        P.arrow(ax, (0, 0), R[:, i], P.COLUMN_COLORS[i], P.column_label(i))

    s = s_length * unit(s_angle)
    q, w = project_onto_cone(R, s)
    ax.plot(*zip(s, q, strict=True), color=P.NNLS_COLOR, lw=1.4, ls=":")
    ax.plot(*q, **{**P.NNLS_STYLE, "linestyle": "none"}, ms=10, zorder=5)
    ax.plot(*s, "o", color=P.INK, ms=5, zorder=6)  # on top: inside the cone, q = s
    ax.annotate("$s$", s, xytext=(7, 5), textcoords="offset points", fontsize=13)
    P.square_limits(ax, radius)
    ax.grid(False)
    ax.set_title(f"$w^*$ = ({w[0]:.2f}, {w[1]:.2f})", fontsize=11)
    if own_figure:
        return fig


def regions_static(theta=60):
    """Four snapshots, with s in each of the four regions."""
    first = 15.0
    angles = [first + theta / 2, first - 50, first + theta + 50, first + theta / 2 + 180]
    titles = ["$s$ inside the cone", "$s$ below $r_1$", "$s$ beyond $r_2$", "$s$ facing away"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, angle, title in zip(axes, angles, titles, strict=True):
        regions_figure(theta, angle, 1.2, ax=ax, n=301)
        ax.set_title(f"{title}\n{ax.get_title()}", fontsize=11)
    P.show(fig)


def regions_widget():
    """Move s around the plane and change the angle between the columns."""
    return widgets.interactive(
        lambda **kw: P.show(regions_figure(**kw)),
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
# Notebook 1: the obtuse-angle test


def obtuse_angle_figure(theta=60):
    """Left: at the closest point q, every point of the cone is at least 90° from s - q.
    Right: at any other point c of the cone, some direction into the cone makes an acute angle
    with s - c, and moving that way gets closer to s."""
    R = two_columns(theta)
    s = 2.2 * unit(-25)
    q, _ = project_onto_cone(R, s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax in axes:
        ax.add_patch(
            Polygon(
                _wedge_outline(R, 6), facecolor=P.REGION_FILLS[(0, 1)], edgecolor="none", zorder=0
            )
        )
        for i in range(2):
            P.arrow(ax, (0, 0), R[:, i], P.COLUMN_COLORS[i], P.column_label(i))
        ax.plot(*s, "o", color=P.INK, ms=7)
        ax.annotate("$s$", s, xytext=(7, -4), textcoords="offset points", fontsize=13)
        P.square_limits(ax, 2.5)
        ax.set_xlim(-0.6, 2.5)
        ax.set_ylim(-1.3, 1.8)
        ax.grid(False)

    ax = axes[0]
    P.arrow(ax, q, s, P.NNLS_COLOR)
    ax.annotate(
        "$s - q$", (q + s) / 2, xytext=(8, 0), textcoords="offset points", color=P.INK, fontsize=12
    )
    for point in [np.zeros(2), 1.8 * R[:, 0], 1.2 * R[:, 1] + 0.3 * R[:, 0]]:
        P.arrow(ax, q, point, P.INK_MUTED)
        angle = np.degrees(
            np.arccos(
                np.dot(s - q, point - q) / (np.linalg.norm(s - q) * np.linalg.norm(point - q))
            )
        )
        ax.annotate(
            f"{angle:.0f}°",
            point,
            xytext=(6, 4),
            textcoords="offset points",
            color=P.INK_SECONDARY,
            fontsize=11,
        )
    ax.plot(*q, **{**P.NNLS_STYLE, "linestyle": "none"}, ms=8, zorder=5)
    ax.annotate("$q$", q, xytext=(-14, 6), textcoords="offset points", fontsize=13)
    ax.set_title("At the closest point $q$: every angle is at least 90°")

    ax = axes[1]
    c = 0.4 * R[:, 0] + 0.6 * R[:, 1]
    P.arrow(ax, c, s, P.INK_SECONDARY)
    ax.annotate(
        "$s - c$", (c + s) / 2, xytext=(8, 4), textcoords="offset points", color=P.INK, fontsize=12
    )
    P.arrow(ax, c, q, P.INK_MUTED)
    angle = np.degrees(
        np.arccos(np.dot(s - c, q - c) / (np.linalg.norm(s - c) * np.linalg.norm(q - c)))
    )
    ax.annotate(
        f"{angle:.0f}°: step this way",
        (c + q) / 2,
        xytext=(4, 10),
        textcoords="offset points",
        ha="left",
        color=P.INK_SECONDARY,
        fontsize=11,
    )
    ax.plot(*c, "o", color=P.INK_SECONDARY, ms=7, zorder=5)
    ax.annotate("$c$", c, xytext=(-14, 6), textcoords="offset points", fontsize=13)
    ax.plot(*q, **{**P.NNLS_STYLE, "linestyle": "none"}, ms=8, zorder=5)
    ax.annotate("$q$", q, xytext=(-14, 6), textcoords="offset points", fontsize=13)
    ax.set_title("At any other point $c$: an acute angle, so not closest")
    return fig
