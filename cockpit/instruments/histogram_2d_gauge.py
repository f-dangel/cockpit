"""Two-dimensional Histogram Gauge."""

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from cockpit.instruments.utils_instruments import (
    _beautify_plot,
    _ticks_formatter,
    check_data,
)


def histogram_2d_gauge(
    self, fig, gridspec, transformation=None, marginals=True, idx=None
):
    """Two-dimensional histogram of the individual gradient and parameter elements.

    This instrument provides a combined histogram of parameter-gradient pairs of
    the network. The values are collected across an entire mini-batch and thus
    captures indvidual gradients as well. The marignal distributions across the
    parameters and gradient values are shown at the top and right respectively.

    The histogram shows the distribution of gradient and parameter elements for
    the last tracked iteration only.

    **Preview**

    .. image:: ../../_static/instrument_previews/Hist2d.png
        :alt: Preview Hist2d Gauge

    **Requires**

    This two dimensional histogram instrument requires data from the
    :class:`~cockpit.quantities.GradHist2d` quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
        transformation (callable): Some map applied to the bin values as a
            transformation for the plot. Defaults to `None` which means no
            transformation.
        marginals (bool): Whether to plot the marginal histograms as well.
        idx (int): Index of parameter whose histogram data should be used.
            If ``None`` (default), uses data of all parameters.
    """
    # Plot
    title_suffix = "(all)" if idx is None else f"(parameter {idx})"
    title = f"Gradient/Parameter Element Histogram {title_suffix}"

    # Check if the required data is available, else skip this instrument
    requires = ["GradHist2d"]

    plot_possible = check_data(self.tracking_data, requires, min_elements=1)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    ax = fig.add_subplot(gridspec)
    ax.set_axis_off()
    ax.set_title(title, fontweight="bold", fontsize="large")

    # Gridspecs (inside gridspec)
    gs = gridspec.subgridspec(3, 3, wspace=0, hspace=0)

    # plot the joint
    if marginals:
        ax_joint = fig.add_subplot(gs[1:, :2])
    else:
        ax_joint = fig.add_subplot(gs[:, :])

    joint_plot_args = {
        "facecolor": self.bg_color_instruments,
        "xlabel": "Parameter Element Value",
        "ylabel": "Gradient Element\nValue",
    }

    df = _get_2d_histogram_data(
        self.tracking_data, transformation=transformation, idx=idx
    )

    cmap = self.alpha_cmap

    sns.heatmap(data=df, cbar=False, cmap=cmap, ax=ax_joint)

    _beautify_plot(ax=ax_joint, **joint_plot_args)

    ax_joint.set_xticklabels(_ticks_formatter(ax_joint.get_xticklabels()))
    ax_joint.set_yticklabels(_ticks_formatter(ax_joint.get_yticklabels()))

    # "Zero lines
    # TODO This assumes that the bins are symmetrical!
    ax_joint.axvline(df.shape[1] / 2, ls="-", color="#ababba", linewidth=1.5, zorder=0)
    ax_joint.axhline(df.shape[0] / 2, ls="-", color="#ababba", linewidth=1.5, zorder=0)

    # plot the marginals
    if marginals:
        ax_xmargin = fig.add_subplot(gs[1:, 2])
        ax_xmargin.set_xscale("log")
        ax_xmargin.get_yaxis().set_visible(False)

        vals, mid_points, bin_size = _get_xmargin_histogram_data(
            self.tracking_data, idx=idx
        )
        ax_xmargin.set_ylim(
            [mid_points[0] - bin_size / 2, mid_points[-1] + bin_size / 2]
        )
        ax_xmargin.barh(
            mid_points, vals, height=bin_size, color=self.primary_color, linewidth=0.1
        )
        ax_xmargin.xaxis.set_minor_locator(ticker.MaxNLocator(3))

        ax_ymargin = fig.add_subplot(gs[0, :2])
        ax_ymargin.set_yscale("log")
        ax_ymargin.get_xaxis().set_visible(False)

        vals, mid_points, bin_size = _get_ymargin_histogram_data(
            self.tracking_data, idx=idx
        )
        ax_ymargin.set_xlim(
            [mid_points[0] - bin_size / 2, mid_points[-1] + bin_size / 2]
        )
        ax_ymargin.bar(
            mid_points,
            vals,
            width=bin_size,
            color=self.primary_color,
            linewidth=0.2,
        )
        ax_ymargin.yaxis.set_minor_locator(ticker.MaxNLocator(3))
        ax_ymargin.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.3g"))


def _default_trafo(array):
    """Default transformation applied to bin counts."""
    return np.log10(array + 1)


def _get_2d_histogram_data(tracking_data, transformation=None, idx=None):
    """Returns the histogram data for the plot.

    Currently we return the bins and values of the last iteration tracked before
    this plot.

    Args:
        tracking_data (pandas.DataFrame): DataFrame holding the tracking data.
        transformation (method): Some map applied to the bin values as a
            transformation for the plot. Use logarithmic transformation per default.
        idx (int): Index of parameter whose histogram data should be used.
            If ``None`` (default), uses data of all parameters.

    Returns:
        DataFrame: DataFrame holding the x and y mid_points and bin values.
    """
    clean_data = tracking_data.GradHist2d.dropna()
    last_step_data = clean_data[clean_data.index[-1]]

    if idx is not None:
        param_key = f"param_{idx}"
        last_step_data = last_step_data[param_key]

    vals = last_step_data["hist"]

    # apply transformation
    if transformation is None:
        transformation = _default_trafo

    vals = transformation(vals)

    x_bins, y_bins = last_step_data["edges"]

    x_mid_points = (x_bins[1:] + x_bins[:-1]) / 2
    y_mid_points = (y_bins[1:] + y_bins[:-1]) / 2

    df = pd.DataFrame(
        data=vals, index=x_mid_points.round(2), columns=y_mid_points.round(2)
    )

    return df


def _get_xmargin_histogram_data(tracking_data, idx=None):
    """Compute histogram data when marginalizing out y-dimension.

    Args:
        tracking_data (pandas.DataFrame): DataFrame holding the tracking data.
        idx (int, optional): Index of parameter whose histogram data should be used.
            If ``None``, uses data of all parameters. Defaults to ``None``.

    Returns:
        vals (numpy.array): Bin counts of one-dimensional histogram when the
            two-dimensional histogram is reduced over the y-dimension.
        mid_points (numpy.array): One-dimensional array containing the center
            points of the histogram bins.
        bin_size (float): Width of a bin.
    """
    clean_data = tracking_data.GradHist2d.dropna()
    last_step_data = clean_data[clean_data.index[-1]]

    if idx is not None:
        param_key = f"param_{idx}"
        last_step_data = last_step_data[param_key]

    vals = last_step_data["hist"].sum(1)
    bins = last_step_data["edges"][0]
    # invert to be consistent with 2d plot
    vals = vals[::-1]

    bin_size = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, bin_size


def _get_ymargin_histogram_data(tracking_data, idx=None):
    """Compute histogram data when marginalizing out x-dimension.

    Args:
        tracking_data (pandas.DataFrame): DataFrame holding the tracking data.
        idx (int, optional): Index of parameter whose histogram data should be used.
            If ``None``, uses data of all parameters. Defaults to ``None``.

    Returns:
        vals (numpy.array): Bin counts of one-dimensional histogram when the
            two-dimensional histogram is reduced over the y-dimension.
        mid_points (numpy.array): One-dimensional array containing the center
            points of the histogram bins.
        bin_size (float): Width of a bin.
    """
    clean_data = tracking_data.GradHist2d.dropna()
    last_step_data = clean_data[clean_data.index[-1]]

    if idx is not None:
        param_key = f"param_{idx}"
        last_step_data = last_step_data[param_key]

    vals = last_step_data["hist"].sum(0)
    bins = last_step_data["edges"][1]

    bin_size = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, bin_size
