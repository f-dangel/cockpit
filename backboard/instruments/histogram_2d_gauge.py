"""Two-dimensional Histogram Gauge."""

import numpy as np
import pandas as pd
import seaborn as sns

from backboard.instruments.utils_instruments import _beautify_plot


def histogram_2d_gauge(self, fig, gridspec, transformation=None):
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
    ax = fig.add_subplot(gridspec)

    plot_args = {
        "title": title,
        "fontweight": "bold",
        "facecolor": "summary",
        "xlabel": "parameter element value",
        "ylabel": "gradient element value",
    }

    df = _get_2d_histogram_data(self.tracking_data, transformation=transformation)

    cmap = self.alpha_cmap

    sns.heatmap(data=df, cbar=False, cmap=cmap)

    _beautify_plot(ax=ax, **plot_args)

    # "Zero lines
    # TODO This assumes that the bins are symmetrical!
    ax.axvline(df.shape[1] / 2, ls="-", color="white", linewidth=1.5, zorder=0)
    ax.axhline(df.shape[0] / 2, ls="-", color="white", linewidth=1.5, zorder=0)

    ax.set_title(title, fontweight="bold", fontsize="large")


def _get_2d_histogram_data(tracking_data, transformation):
    """Returns the histogram data for the plot.

    Currently we return the bins and values of the last iteration tracked before
    this plot.

    Args:
        tracking_data (pandas.DataFrame): DataFrame holding the tracking data.
        transformation (method): Some map applied to the bin values as a
            transformation for the plot.
    """
    data = tracking_data[["x_edges", "y_edges", "hist_2d"]].dropna().tail(1)

    vals = np.array(data.hist_2d.to_numpy()[0])

    # apply transformation
    if transformation is not None:
        vals = transformation(vals)

    x_bins = np.array(data.x_edges.to_numpy()[0])
    y_bins = np.array(data.y_edges.to_numpy()[0])

    x_mid_points = (x_bins[1:] + x_bins[:-1]) / 2
    y_mid_points = (y_bins[1:] + y_bins[:-1]) / 2

    df = pd.DataFrame(
        data=vals, index=x_mid_points.round(2), columns=y_mid_points.round(2)
    )

    return df
