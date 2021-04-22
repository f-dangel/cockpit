"""Gradient Norm Gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot
from cockpit.quantities.utils_quantities import _root_sum_of_squares


def grad_norm_gauge(self, fig, gridspec):
    """Showing the gradient norm versus iteration.

    If the training gets stuck, due to a small
    :class:`~cockpit.quantities.UpdateSize` it can be the result of both a badly
    chosen learning rate, or from a flat plateau in the loss landscape.
    This instrument shows the gradient norm at each iteration, overlayed with an
    exponentially weighted average, and can thus distinguish these two cases.

    **Preview**

    .. image:: ../../_static/instrument_previews/GradientNorm.png
        :alt: Preview GradientNorm Gauge

    **Requires**

    The gradient norm instrument requires data from the
    :class:`~cockpit.quantities.GradNorm` quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Gradient Norm"

    # Check if the required data is available, else skip this instrument
    requires = ["GradNorm"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    # Compute
    self.tracking_data["GradNorm_all"] = self.tracking_data.GradNorm.map(
        lambda x: _root_sum_of_squares(x) if type(x) == list else x
    )

    plot_args = {
        "x": "iteration",
        "y": "GradNorm_all",
        "data": self.tracking_data,
        "x_scale": "symlog" if self.show_log_iter else "linear",
        "y_scale": "linear",
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
