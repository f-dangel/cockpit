"""Alpha Gauge."""


def alpha_gauge(self, fig, gridspec):
    """Showing a distribution of the alpha values since the last plot.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed.
    """
    ax = fig.add_subplot(gridspec)
    ax.set_title("Alpha Gauge")
