"""Max EV Gauge."""

import warnings

from matplotlib import ticker

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def max_ev_gauge(self, fig, gridspec):
    """Showing the largest eigenvalue of the Hessian versus iteration.

    The largest eigenvalue of the Hessian indicates the loss surface's sharpest
    valley. Together with the :func:`~cockpit.instruments.trace_gauge()`, which
    provides a notion of "average curvature", it can help understand the "average
    condition number" of the loss landscape at the current point. The instrument
    shows the largest eigenvalue of the Hessian versus iteration, overlayed with
    an exponentially weighted average.

    **Preview**

    .. image:: ../../_static/instrument_previews/HessMaxEV.png
        :alt: Preview HessMaxEV Gauge

    **Requires**

    The trace instrument requires data from the :class:`~cockpit.quantities.HessMaxEv`
    quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed.
    """
    # Plot Trace vs iteration
    title = "Max Eigenvalue"

    # Check if the required data is available, else skip this instrument
    requires = ["HessMaxEV"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    plot_args = {
        "x": "iteration",
        "y": "HessMaxEV",
        "data": self.tracking_data,
        "x_scale": "symlog" if self.show_log_iter else "linear",
        "y_scale": "log",
        "cmap": self.cmap,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "title": title,
        "xlim": "tight",
        "ylim": None,
        "fontweight": "bold",
        "facecolor": self.bg_color_instruments,
    }
    # part that should be plotted
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)

    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.2g"))
