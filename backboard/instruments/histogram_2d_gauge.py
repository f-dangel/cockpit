"""Two-dimensional Histogram Gauge."""

import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from backboard.instruments.utils_instruments import _beautify_plot, check_data


def histogram_2d_gauge(self, fig, gridspec, transformation=None):
    """Two-dimensional histogram of the individual gradient and parameter elements.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
        transformation (method): Some map applied to the bin values as a
            transformation for the plot. Defaults to `None` which means no
            transformation.
    """
    # Plot
    title = "Gradient/Parameter Element Histogram"

    # Check if the required data is available, else skip this instrument
    requires = ["x_edges", "y_edges", "hist_2d"]
    plot_possible = check_data(self.tracking_data, requires, min_elements=1)
    if not plot_possible:
        warnings.warn(
            "Couldn't get the required data for the " + title + " instrument",
            stacklevel=1,
        )
        return

    ax = fig.add_subplot(gridspec)

    plot_args = {
        "title": title,
        "fontweight": "bold",
        "facecolor": self.bg_color_instruments,
        "xlabel": "parameter element value",
        "ylabel": "gradient element value",
    }

    df = _get_2d_histogram_data(self.tracking_data, transformation=transformation)

    cmap = self.alpha_cmap

    sns.heatmap(data=df, cbar=False, cmap=cmap, ax=ax)

    _beautify_plot(ax=ax, **plot_args)

    # "Zero lines
    # TODO This assumes that the bins are symmetrical!
    ax.axvline(df.shape[1] / 2, ls="-", color="#ababba", linewidth=1.5, zorder=0)
    ax.axhline(df.shape[0] / 2, ls="-", color="#ababba", linewidth=1.5, zorder=0)

    ax.set_title(title, fontweight="bold", fontsize="large")


def histogram_2d_gauge_marginal(self, fig, gridspec, transformation=None):
    """Twp-dimensional histogram of the individual gradient and parameter elements.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
        transformation (method): Some map applied to the bin values as a
            transformation for the plot. Defaults to `None` which means no
            transformation.
    """
    # Plot
    title = "Gradient/Parameter Element Histogram"

    # Check if the required data is available, else skip this instrument
    requires = ["x_edges", "y_edges", "hist_2d"]
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

    plot_args = {
        "facecolor": self.bg_color_instruments,
        "xlabel": "parameter element value",
        "ylabel": "gradient element value",
    }

    df = _get_2d_histogram_data(self.tracking_data, transformation=transformation)

    cmap = self.alpha_cmap

    # Gridspecs (inside gridspec)
    gs = gridspec.subgridspec(3, 3, wspace=0, hspace=0)

    ax_joint = fig.add_subplot(gs[1:, :2])
    sns.heatmap(data=df, cbar=False, cmap=cmap, ax=ax_joint)
    _beautify_plot(ax=ax_joint, **plot_args)

    # "Zero lines
    # TODO This assumes that the bins are symmetrical!
    ax_joint.axvline(df.shape[1] / 2, ls="-", color="#ababba", linewidth=1.5, zorder=0)
    ax_joint.axhline(df.shape[0] / 2, ls="-", color="#ababba", linewidth=1.5, zorder=0)

    # plot_args.pop("xlabel")
    # plot_args.pop("ylabel")

    ax_margin_x = fig.add_subplot(gs[1:, 2])
    ax_margin_x.set_xscale("log")
    ax_margin_x.get_yaxis().set_visible(False)
    # TODO Use exactly the same y limits as ax_joint
    vals, mid_points, height = _get_marginal_histogram_data_x(self.tracking_data)
    ax_margin_x.barh(
        mid_points, vals, height=height, color=self.primary_color, linewidth=0.1
    )
    # _beautify_plot(ax_margin_x, **plot_args)

    ax_margin_y = fig.add_subplot(gs[0, :2])
    ax_margin_y.set_yscale("log")
    ax_margin_y.get_xaxis().set_visible(False)
    # TODO Use exactly the same x limits as ax_joint
    vals, mid_points, width = _get_marginal_histogram_data_y(self.tracking_data)
    ax_margin_y.bar(
        mid_points, vals, width=width, color=self.primary_color, linewidth=0.2
    )
    # _beautify_plot(ax_margin_y, **plot_args)


# use marginal plot
WITH_MARGINALS = True
if WITH_MARGINALS:
    histogram_2d_gauge = histogram_2d_gauge_marginal


def _default_trafo(array):
    """Default transformation applied to bin counts."""
    return np.log10(array + 1)


def _get_2d_histogram_data(tracking_data, transformation=None):
    """Returns the histogram data for the plot.

    Currently we return the bins and values of the last iteration tracked before
    this plot.

    Args:
        tracking_data (pandas.DataFrame): DataFrame holding the tracking data.
        transformation (method): Some map applied to the bin values as a
            transformation for the plot. Use logarithmic transformation per default.
    """
    data = tracking_data[["x_edges", "y_edges", "hist_2d"]].dropna().tail(1)

    vals = np.array(data.hist_2d.to_numpy()[0])

    # apply transformation
    if transformation is None:
        transformation = _default_trafo

    vals = transformation(vals)

    x_bins = np.array(data.x_edges.to_numpy()[0])
    y_bins = np.array(data.y_edges.to_numpy()[0])

    x_mid_points = (x_bins[1:] + x_bins[:-1]) / 2
    y_mid_points = (y_bins[1:] + y_bins[:-1]) / 2

    df = pd.DataFrame(
        data=vals, index=x_mid_points.round(2), columns=y_mid_points.round(2)
    )

    return df


def _get_marginal_histogram_data_x(tracking_data):
    data = tracking_data[["x_edges", "hist_2d"]].dropna().tail(1)

    vals = np.array(data.hist_2d.to_numpy()[0]).sum(1)
    bins = np.array(data.x_edges.to_numpy()[0])

    width = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, width


def _get_marginal_histogram_data_y(tracking_data):
    data = tracking_data[["y_edges", "hist_2d"]].dropna().tail(1)

    vals = np.array(data.hist_2d.to_numpy()[0]).sum(0)
    bins = np.array(data.y_edges.to_numpy()[0])

    width = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, width
