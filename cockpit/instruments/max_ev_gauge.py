"""Max EV Gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def max_ev_gauge(self, fig, gridspec):
    """Showing the largest eigenvalue of the Hessian versus iteration.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
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
