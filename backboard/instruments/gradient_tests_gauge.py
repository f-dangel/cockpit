"""Gradient Tests Gauge."""

# from backboard.instruments.utils_instruments import create_basic_plot


def gradient_tests_gauge(self, fig, gridspec):
    """Gauge, showing the the status of several gradient tests.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    # Plot
    title = "Gradient Tests"
    ax = fig.add_subplot(gridspec)

    ax.set_title(title, fontweight="bold", fontsize="large")
