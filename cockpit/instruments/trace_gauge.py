"""Trace Gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def trace_gauge(self, fig, gridspec):
    """Trace gauge, showing the trace of the Hessian versus iteration.

    The trace of the hessian is the sum of its eigenvalues and thus can indicate
    the overall or average curvature of the loss landscape at the current point.
    Increasing values for the trace indicate a steeper curvature, for example, a
    narrower valley. This instrument shows the trace versus iteration, overlayed
    with an exponentially weighted average.

    **Preview**

    .. image:: ../../_static/instrument_previews/HessTrace.png
        :alt: Preview HessTrace Gauge

    **Requires**

    The trace instrument requires data from the :class:`~cockpit.quantities.HessTrace`
    quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Trace"

    # Check if the required data is available, else skip this instrument
    requires = ["HessTrace"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    # Compute
    self.tracking_data["HessTrace_all"] = self.tracking_data.HessTrace.map(
        lambda x: sum(x) if type(x) == list else x
    )

    plot_args = {
        "x": "iteration",
        "y": "HessTrace_all",
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
