"""MWE how plotting of gradient tests could look like.

TODO Integrate into plotter
"""

import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy


def create_fake_data(steps):
    """Create fake data for testing."""
    random.seed(0)
    log = {}

    for s in range(steps):
        step_log = {
            "norm_test_radius": 0.8 * random.random(),
            "inner_product_test_width": 0.8 * random.random(),
            "orthogonality_test_width": 0.8 * random.random(),
        }
        log[s] = step_log

    return log


def gradient_tests(self, fig, gridspec):
    """Plot cockpit visualization of gradient tests."""
    if self is not None or fig is not None or gridspec is not None:
        raise NotImplementedError("Need to integrate arguments")

    steps = 30
    log = create_fake_data(steps)

    fig = plt.figure(constrained_layout=True)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # tiling
    gs = fig.add_gridspec(3, 3)

    fig_all = fig.add_subplot(gs[1:, 1:])
    fig_norm = fig.add_subplot(gs[1, 0])
    fig_inner = fig.add_subplot(gs[2, 0])
    fig_ortho = fig.add_subplot(gs[0, 2])

    _format(fig_all, fig_norm, fig_inner, fig_ortho)
    _plot(fig_all, fig_norm, fig_inner, fig_ortho, log)

    return fig


def _format(fig_all, fig_norm, fig_inner, fig_ortho):
    """Format axes of all subplots."""
    fig_all.yaxis.tick_right()
    fig_all.set_xlim([-1, 1])
    fig_all.set_ylim([0, 2])

    fig_all.set_axisbelow(True)
    fig_all.grid(ls="--")
    fig_all.plot(0, 1, color="black", marker="+", markersize=18, markeredgewidth=4)

    fig_norm.set_ylabel("norm")
    fig_norm.set_ylim([0, 1])
    fig_norm.xaxis.tick_top()

    fig_inner.set_ylabel("inner")
    fig_inner.set_ylim([0, 1])
    fig_inner.invert_yaxis()

    fig_ortho.set_title("ortho")
    fig_ortho.xaxis.tick_top()
    fig_ortho.yaxis.tick_right()
    fig_ortho.set_xlim([0, 1])
    fig_ortho.invert_yaxis()


def _plot(fig_all, fig_norm, fig_inner, fig_ortho, log):
    """Plot data."""
    # data extraction
    steps = len(log.keys())
    steps_array = numpy.arange(0, steps)
    norm_test_radii = [log[s]["norm_test_radius"] for s in range(steps)]
    inner_product_test_widths = [
        log[s]["inner_product_test_width"] for s in range(steps)
    ]
    orthogonality_test_widths = [
        log[s]["orthogonality_test_width"] for s in range(steps)
    ]

    # plot norm test
    fig_all.add_artist(
        plt.Circle((0, 1), norm_test_radii[-1], color="blue", fill=False)
    )
    fig_all.add_artist(plt.Circle((0, 1), norm_test_radii[-1], color="blue", alpha=0.4))
    fig_all.plot(
        [-1, -1],
        [1, 1 + norm_test_radii[-1]],
        color="blue",
        linewidth=6,
    )

    fig_norm.fill_between(steps_array, norm_test_radii, color="blue", alpha=0.4)
    fig_norm.plot(steps_array, norm_test_radii, color="blue")

    # plot inner product test
    fig_all.axhspan(
        1 - inner_product_test_widths[-1],
        1 + inner_product_test_widths[-1],
        color="red",
        alpha=0.4,
    )
    fig_all.axhspan(
        1 - inner_product_test_widths[-1],
        1 + inner_product_test_widths[-1],
        color="red",
        fill=False,
    )
    fig_all.plot(
        [-1, -1],
        [1 - inner_product_test_widths[-1], 1],
        color="red",
        linewidth=6,
    )

    fig_inner.fill_between(
        steps_array, inner_product_test_widths, color="red", alpha=0.4
    )
    fig_inner.plot(steps_array, inner_product_test_widths, color="red")

    # plot orthogonality test
    fig_all.axvspan(
        -orthogonality_test_widths[-1],
        orthogonality_test_widths[-1],
        color="green",
        alpha=0.4,
    )
    fig_all.axvspan(
        -orthogonality_test_widths[-1],
        orthogonality_test_widths[-1],
        color="green",
        fill=False,
    )
    fig_all.plot(
        [0, orthogonality_test_widths[-1]],
        [2, 2],
        color="green",
        linewidth=6,
    )

    fig_ortho.plot(orthogonality_test_widths, steps_array, color="green")

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
    fig_ortho.add_patch(patch)


if __name__ == "__main__":
    # TODO Integrate
    self, fig, gridspec = None, None, None

    gradient_tests(self, fig, gridspec)

    plt.show(block=False)
    plt.waitforbuttonpress()
