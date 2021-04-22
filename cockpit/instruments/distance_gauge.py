"""Distance Gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot
from cockpit.quantities.utils_quantities import _root_sum_of_squares


def distance_gauge(self, fig, gridspec):
    """Distance gauge showing two different quantities related to distance.

    This instruments shows two quantities at once. Firstly, the :math:`L_2`-distance
    of the current parameters to their initialization. This describes the total distance
    that the optimization trajectory "has traveled so far" and can be seen via the
    blue-to-green dots (and the left y-axis).

    Secondly, the update sizes of individual steps are shown via the yellow-to-blue
    dots (and the right y-axis). It measure the distance that a single parameter
    update covers.

    Both quantities are overlayed with an exponentially weighted average.

    .. image:: ../../_static/instrument_previews/Distances.png
        :alt: Preview Distances Gauge

    **Requires**

    The distance instrument requires data from both, the
    :class:`~cockpit.quantities.UpdateSize` and the
    :class:`~cockpit.quantities.Distance` quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Distance"

    # Check if the required data is available, else skip this instrument
    requires = ["Distance", "UpdateSize"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    # Compute
    self.tracking_data["Distance_all"] = self.tracking_data.Distance.map(
        lambda x: _root_sum_of_squares(x) if type(x) == list else x
    )
    self.tracking_data["UpdateSize_all"] = self.tracking_data.UpdateSize.map(
        lambda x: _root_sum_of_squares(x) if type(x) == list else x
    )

    plot_args = {
        "x": "iteration",
        "y": "Distance_all",
        "data": self.tracking_data,
        "y_scale": "linear",
        "x_scale": "symlog" if self.show_log_iter else "linear",
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
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)

    ax2 = ax.twinx()
    plot_args = {
        "x": "iteration",
        "y": "UpdateSize_all",
        "data": self.tracking_data,
        "y_scale": "linear",
        "x_scale": "symlog" if self.show_log_iter else "linear",
        "cmap": self.cmap.reversed(),
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2.reversed(),
        "xlim": "tight",
        "ylim": None,
        "marker": ",",
    }
    create_basic_plot(**plot_args, ax=ax2)
