"""One-dimensional Histogram Gauge."""

import warnings

from cockpit.instruments.utils_instruments import _beautify_plot, check_data


def histogram_1d_gauge(self, fig, gridspec, y_scale="log"):
    """One-dimensional histogram of the individual gradient elements.

    This instrument provides a histogram of the gradient element values across all
    individual gradients in a mini-batch. The histogram shows the distribution for
    the last tracked iteration only.

    **Preview**

    .. image:: ../../_static/instrument_previews/Hist1d.png
        :alt: Preview Hist1d Gauge

    **Requires**

    This two dimensional histogram instrument requires data from the
    :class:`~cockpit.quantities.GradHist1d` quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
        y_scale (str, optional): Scale of the y-axis. Defaults to "log".
    """
    # Plot
    title = "Gradient Element Histogram"

    # Check if the required data is available, else skip this instrument
    requires = ["GradHist1d"]
    plot_possible = check_data(self.tracking_data, requires, min_elements=1)
    if not plot_possible:
        if self.debug:
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
        "xlabel": "Gradient Element Value",
        "ylabel": "Frequency",
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

    Returns:
        list: Bins of the histogram.
        list: Mid points of the bins.
        list: Width of the bins.
    """
    clean_data = tracking_data.GradHist1d.dropna()
    last_step_data = clean_data[clean_data.index[-1]]

    vals = last_step_data["hist"]
    bins = last_step_data["edges"]

    width = bins[1] - bins[0]

    mid_points = (bins[1:] + bins[:-1]) / 2

    return vals, mid_points, width
