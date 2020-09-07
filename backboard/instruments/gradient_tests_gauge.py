"""Gradient Tests Gauge."""

import matplotlib as mpl
import matplotlib.pyplot as plt


def gradient_tests_gauge(self, fig, gridspec):
    """Gauge, showing the the status of several gradient tests.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    # Plot
    title = "Gradient Tests"

    ax = fig.add_subplot(gridspec)
    ax.set_title(title, fontweight="bold", fontsize="large")
    ax.set_axis_off()

    # Gridspecs (inside gridspec)
    gs = gridspec.subgridspec(3, 3, wspace=0, hspace=0)

    ax_all = fig.add_subplot(gs[1:, 1:])
    ax_norm = fig.add_subplot(gs[1, 0])
    ax_inner = fig.add_subplot(gs[2, 0])
    ax_ortho = fig.add_subplot(gs[0, 2])

    _format(ax_all, ax_norm, ax_inner, ax_ortho)
    _plot(self, ax_all, ax_norm, ax_inner, ax_ortho)


def _format(ax_all, ax_norm, ax_inner, ax_ortho):
    """Format axes of all subplots."""
    ax_all.yaxis.tick_right()
    ax_all.set_xlim([-1, 1])
    ax_all.set_ylim([0, 2])

    ax_all.set_axisbelow(True)
    ax_all.grid(ls="--")
    ax_all.plot(0, 1, color="black", marker="+", markersize=18, markeredgewidth=4)

    ax_norm.set_ylabel("norm")
    ax_norm.set_ylim([0, 1])
    ax_norm.xaxis.tick_top()

    ax_inner.set_ylabel("inner")
    ax_inner.set_ylim([0, 1])
    ax_inner.invert_yaxis()

    ax_ortho.set_title("ortho")
    ax_ortho.xaxis.tick_top()
    ax_ortho.yaxis.tick_right()
    ax_ortho.set_xlim([0, 1])
    ax_ortho.invert_yaxis()


def _plot(self, ax_all, ax_norm, ax_inner, ax_ortho):
    """Plot data."""
    # data extraction
    log = self.tracking_data[
        ["iteration", "inner_product_test", "norm_test", "orthogonality_test"]
    ].dropna()

    steps_array = log.iteration.tolist()
    norm_test_radii = log.norm_test.tolist()
    inner_product_test_widths = log.inner_product_test.tolist()
    orthogonality_test_widths = log.orthogonality_test.tolist()

    # plot norm test
    ax_all.add_artist(plt.Circle((0, 1), norm_test_radii[-1], color="blue", fill=False))
    ax_all.add_artist(plt.Circle((0, 1), norm_test_radii[-1], color="blue", alpha=0.4))
    ax_all.plot(
        [-1, -1],
        [1, 1 + norm_test_radii[-1]],
        color="blue",
        linewidth=6,
    )

    ax_norm.fill_between(steps_array, norm_test_radii, color="blue", alpha=0.4)
    ax_norm.plot(steps_array, norm_test_radii, color="blue")

    # plot inner product test
    ax_all.axhspan(
        1 - inner_product_test_widths[-1],
        1 + inner_product_test_widths[-1],
        color="red",
        alpha=0.4,
    )
    ax_all.axhspan(
        1 - inner_product_test_widths[-1],
        1 + inner_product_test_widths[-1],
        color="red",
        fill=False,
    )
    ax_all.plot(
        [-1, -1],
        [1 - inner_product_test_widths[-1], 1],
        color="red",
        linewidth=6,
    )

    ax_inner.fill_between(
        steps_array, inner_product_test_widths, color="red", alpha=0.4
    )
    ax_inner.plot(steps_array, inner_product_test_widths, color="red")

    # plot orthogonality test
    ax_all.axvspan(
        -orthogonality_test_widths[-1],
        orthogonality_test_widths[-1],
        color="green",
        alpha=0.4,
    )
    ax_all.axvspan(
        -orthogonality_test_widths[-1],
        orthogonality_test_widths[-1],
        color="green",
        fill=False,
    )
    ax_all.plot(
        [0, orthogonality_test_widths[-1]],
        [2, 2],
        color="green",
        linewidth=6,
    )

    ax_ortho.plot(orthogonality_test_widths, steps_array, color="green")

    # workaround to fill between curve and y axis
    ortho_vertices = (
        [(0, 0)]
        + [(x, y) for x, y in zip(orthogonality_test_widths, steps_array)]
        + [(0, steps_array[-1])]
    )
    codes = [mpl.path.Path.LINETO for v in ortho_vertices]
    codes[0] = mpl.path.Path.MOVETO

    path = mpl.path.Path(ortho_vertices, codes)
    patch = mpl.patches.PathPatch(path, facecolor="green", alpha=0.4, lw=0)
    ax_ortho.add_patch(patch)
