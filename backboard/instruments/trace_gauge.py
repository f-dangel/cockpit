"""Trace Gauge."""

from backboard.instruments.utils_instruments import create_basic_plot


def trace_gauge(self, fig, gridspec):
    """Trace gauge, showing the trace of the Hessian versus iteration.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Trace"

    # Compute
    self.tracking_data["trace_all"] = self.tracking_data.trace.map(
        lambda x: sum(x) if type(x) == list else x
    )

    plot_args = {
        "x": "iteration",
        "y": "trace_all",
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
