"""One-dimensional Histogram Gauge."""

import warnings

import numpy as np

from cockpit.instruments.utils_instruments import _beautify_plot, check_data


def histogram_1d_gauge(self, fig, gridspec, y_scale="log"):
    """One-dimensional histogram of the individual gradient elements.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
        y_scale (str, optional): Scale of the y-axis. Defaults to "log".
    """
    # Plot
    title = "Gradient Element Histogram"

    # Check if the required data is available, else skip this instrument
    requires = ["edges", "hist_1d"]
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
        "xlabel": "gradient element value",
        "ylabel": "frequency",
        "y_scale": y_scale,
    }

    vals, mid_points, width = _get_histogram_data(self.tracking_data)

    ax.bar(mid_points, vals, width=width, color=self.primary_color)

    _beautify_plot(ax=ax, **plot_args)

    ax.set_title(title, fontweight="bold", fontsize="large")


def _get_histogram_data(tracking_data):
    """Returns the histogram data for the plot.

    Currently we return the bins and values of the last iteration tracked before
    this plot.

    Args:
        tracking_data (pandas.DataFrame): DataFrame holding the tracking data.
    """
    data = tracking_data[["edges", "hist_1d"]].dropna().tail(1)

    vals = np.array(data.hist_1d.to_numpy()[0])
    bins = np.array(data.edges.to_numpy()[0])

    width = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, width
