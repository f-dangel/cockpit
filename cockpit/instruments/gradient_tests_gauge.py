"""Gradient Tests Gauge."""

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker

from cockpit.instruments.utils_instruments import check_data


def gradient_tests_gauge(self, fig, gridspec):
    """Gauge, showing the the status of several gradient tests.

    All three gradient tests (the norm test, the inner product test, and the
    orthogonality test) indicate how strongly individual gradients in a mini-batch
    scatter around the mean gradient. This information can be used to adapt the
    batch size whenever the information becomes to noisy, as indicated by large
    values.

    The central plot visualizes all three tests in different colors. Each area shows
    how far the individual gradients scatter. The smaller plots show their evolution
    over time.

    **Preview**

    .. image:: ../../_static/instrument_previews/GradientTests.png
        :alt: Preview GradientTests Gauge

    **Requires**

    The gradient test instrument requires data from all three gradient test quantities,
    namely the :class:`~cockpit.quantities.InnerTest`,
    :class:`~cockpit.quantities.NormTest`, and :class:`~cockpit.quantities.OrthoTest`
    quantity classes.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot
    title = "Gradient Tests"

    # Check if the required data is available, else skip this instrument
    requires = ["iteration", "InnerTest", "NormTest", "OrthoTest"]
    plot_possible = check_data(self.tracking_data, requires, min_elements=1)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    ax = fig.add_subplot(gridspec)
    ax.set_title(title, fontweight="bold", fontsize="large")
    ax.set_axis_off()

    # Gridspecs (inside gridspec)
    gs = gridspec.subgridspec(3, 3, wspace=0.05, hspace=0.1)

    ax_all = fig.add_subplot(gs[1:, 1:])
    ax_norm = fig.add_subplot(gs[1, 0])
    ax_inner = fig.add_subplot(gs[2, 0])
    ax_ortho = fig.add_subplot(gs[0, 2])

    _format(self, ax_all, ax_norm, ax_inner, ax_ortho)
    _plot(self, ax_all, ax_norm, ax_inner, ax_ortho)


def _format(self, ax_all, ax_norm, ax_inner, ax_ortho):
    """Format axes of all subplots."""
    iter_scale = "symlog" if self.show_log_iter else "linear"

    # area around cross
    w = 2
    ax_all.yaxis.tick_right()
    ax_all.set_xlim([-w, w])
    ax_all.set_xscale("symlog", linthresh=0.1)
    ax_all.set_ylim([0 - w, 0 + w])
    ax_all.set_yscale("symlog", linthresh=0.1)

    ax_all.set_axisbelow(True)
    ax_all.grid(ls="--")
    ax_all.plot(0, 0, color="black", marker="+", markersize=18, markeredgewidth=4)
    ax_all.set_facecolor(self.bg_color_instruments)

    ax_norm.set_ylabel("Norm")
    ax_norm.set_yscale("log")
    ax_norm.xaxis.tick_top()
    ax_norm.set_facecolor(self.bg_color_instruments)
    ax_norm.set_xscale(iter_scale)
    ax_norm.yaxis.set_minor_locator(ticker.MaxNLocator(3))
    ax_norm.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.2g"))

    ax_inner.set_ylabel("Inner")
    ax_inner.set_yscale("log")
    ax_inner.invert_yaxis()
    ax_inner.set_facecolor(self.bg_color_instruments)
    ax_inner.set_xscale(iter_scale)
    ax_inner.yaxis.set_minor_locator(ticker.MaxNLocator(3))
    ax_inner.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.2g"))

    ax_ortho.set_title("Ortho")
    ax_ortho.xaxis.tick_top()
    ax_ortho.yaxis.tick_right()
    ax_ortho.set_xscale("log")
    ax_ortho.invert_yaxis()
    ax_ortho.set_yscale(iter_scale)
    ax_ortho.set_facecolor(self.bg_color_instruments)
    ax_ortho.xaxis.set_minor_locator(ticker.MaxNLocator(2))
    ax_ortho.xaxis.set_minor_formatter(ticker.FormatStrFormatter("%.2g"))


def _plot(self, ax_all, ax_norm, ax_inner, ax_ortho):
    """Plot data."""
    # data extraction
    log = self.tracking_data[
        ["iteration", "InnerTest", "NormTest", "OrthoTest"]
    ].dropna()

    steps_array = log.iteration.tolist()
    norm_test_radii = log.NormTest.tolist()
    inner_product_test_widths = log.InnerTest.tolist()
    orthogonality_test_widths = log.OrthoTest.tolist()

    # plot norm test
    ax_all.add_artist(
        plt.Circle((0, 0), norm_test_radii[-1], color=self.primary_color, fill=False)
    )
    ax_all.add_artist(
        plt.Circle((0, 0), norm_test_radii[-1], color=self.primary_color, alpha=0.5)
    )

    ax_norm.fill_between(
        steps_array, norm_test_radii, color=self.primary_color, alpha=0.5
    )
    ax_norm.plot(steps_array, norm_test_radii, color=self.primary_color)

    # plot inner product test
    ax_all.axhspan(
        -inner_product_test_widths[-1],
        inner_product_test_widths[-1],
        color=self.secondary_color,
        alpha=0.5,
    )
    ax_all.axhspan(
        -inner_product_test_widths[-1],
        inner_product_test_widths[-1],
        color=self.secondary_color,
        fill=False,
    )

    ax_inner.fill_between(
        steps_array, inner_product_test_widths, color=self.secondary_color, alpha=0.5
    )
    ax_inner.plot(steps_array, inner_product_test_widths, color=self.secondary_color)

    # plot orthogonality test
    ax_all.axvspan(
        -orthogonality_test_widths[-1],
        orthogonality_test_widths[-1],
        color=self.tertiary_color,
        alpha=0.5,
    )
    ax_all.axvspan(
        -orthogonality_test_widths[-1],
        orthogonality_test_widths[-1],
        color=self.tertiary_color,
        fill=False,
    )

    ax_ortho.plot(orthogonality_test_widths, steps_array, color=self.tertiary_color)

    # workaround to fill between curve and y axis
    ortho_vertices = (
        [(0, 0)]
        + [(x, y) for x, y in zip(orthogonality_test_widths, steps_array)]
        + [(0, steps_array[-1])]
    )
    codes = [mpl.path.Path.LINETO for v in ortho_vertices]
    codes[0] = mpl.path.Path.MOVETO

    path = mpl.path.Path(ortho_vertices, codes)
    patch = mpl.patches.PathPatch(path, facecolor=self.tertiary_color, alpha=0.5, lw=0)
    ax_ortho.add_patch(patch)
