"""Two-dimensional Histogram Gauge."""

import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from cockpit.instruments.utils_instruments import _beautify_plot, check_data


def histogram_2d_gauge(
    self, fig, gridspec, transformation=None, marginals=True, idx=None
):
    """Two-dimensional histogram of the individual gradient and parameter elements.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
        transformation (method): Some map applied to the bin values as a
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
        "ylabel": "Gradient Element Value",
    }

    df = _get_2d_histogram_data(
        self.tracking_data, transformation=transformation, idx=idx
    )

    cmap = self.alpha_cmap

    sns.heatmap(data=df, cbar=False, cmap=cmap, ax=ax_joint)

    _beautify_plot(ax=ax_joint, **joint_plot_args)

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
    """
    last_step_data = tracking_data.GradHist2d.dropna()[tracking_data.index[-1]]

    key_prefix = "" if idx is None else f"param_{idx}_"
    x_key = key_prefix + "x_edges"
    y_key = key_prefix + "y_edges"
    hist_key = key_prefix + "hist_2d"

    vals = np.array(last_step_data[hist_key])

    # apply transformation
    if transformation is None:
        transformation = _default_trafo

    vals = transformation(vals)

    x_bins = np.array(last_step_data[x_key])
    y_bins = np.array(last_step_data[y_key])

    x_mid_points = (x_bins[1:] + x_bins[:-1]) / 2
    y_mid_points = (y_bins[1:] + y_bins[:-1]) / 2

    df = pd.DataFrame(
        data=vals, index=x_mid_points.round(2), columns=y_mid_points.round(2)
    )

    return df


def _get_xmargin_histogram_data(tracking_data, idx=None):
    """Compute histogram data when marginalizing out y-dimension.

    Returns:
        vals (numpy.array): Bin counts of one-dimensional histogram when the
            two-dimensional histogram is reduced over the y-dimension.
        mid_points (numpy.array): One-dimensional array containing the center
            points of the histogram bins.
        bin_size (float): Width of a bin.
        idx (int): Index of parameter whose histogram data should be used.
            If ``None`` (default), uses data of all parameters.
    """
    key_prefix = "" if idx is None else f"param_{idx}_"
    x_key = key_prefix + "x_edges"
    hist_key = key_prefix + "hist_2d"

    last_step_data = tracking_data.GradHist2d.dropna()[tracking_data.index[-1]]

    vals = np.array(last_step_data[hist_key]).sum(1)
    bins = np.array(last_step_data[x_key])
    # invert to be consistent with 2d plot
    vals = vals[::-1]

    bin_size = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, bin_size


def _get_ymargin_histogram_data(tracking_data, idx=None):
    """Compute histogram data when marginalizing out x-dimension.

    Returns:
        vals (numpy.array): Bin counts of one-dimensional histogram when the
            two-dimensional histogram is reduced over the x-dimension.
        mid_points (numpy.array): One-dimensional array containing the center
            points of the histogram bins.
        bin_size (float): Width of a bin.
        idx (int): Index of parameter whose histogram data should be used.
            If ``None`` (default), uses data of all parameters.
    """
    key_prefix = "" if idx is None else f"param_{idx}_"
    y_key = key_prefix + "y_edges"
    hist_key = key_prefix + "hist_2d"

    last_step_data = tracking_data.GradHist2d.dropna()[tracking_data.index[-1]]

    vals = np.array(last_step_data[hist_key]).sum(0)
    bins = np.array(last_step_data[y_key])

    bin_size = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, bin_size
