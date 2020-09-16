"""Distance Gauge."""

import warnings

from backboard.instruments.utils_instruments import check_data, create_basic_plot
from backboard.quantities.utils_quantities import _root_sum_of_squares


def distance_gauge(self, fig, gridspec):
    """Showing the parameter L2-distance to the initialization versus iteration.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Distance"

    # Check if the required data is available, else skip this instrument
    requires = ["d2init"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        warnings.warn(
            "Couldn't get the required data for the " + title + " instrument",
            stacklevel=1,
        )
        return

    # Compute
    self.tracking_data["d2init_all"] = self.tracking_data.d2init.map(
        lambda x: _root_sum_of_squares(x) if type(x) == list else x
    )

    plot_args = {
        "x": "iteration",
        "y": "d2init_all",
        "data": self.tracking_data,
        "y_scale": "linear",
        "cmap": self.cmap,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "title": title,
        "xlim": "tight",
        "ylim": None,
        "fontweight": "bold",
        "facecolor": "summary",
    }
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)
