"""Hyperparameter Gauge."""

import seaborn as sns

from .utils_instruments import _add_last_value_to_legend, _beautify_plot


def hyperparameter_gauge(self, fig, gridspec):
    """Hyperparameter gauge, currently showing the learning rate over time.

    Args:
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    ax = fig.add_subplot(gridspec)

    # Plot Settings
    plot_args = {
        "x": "iteration",
        "y": "learning_rate",
        "data": self.epoch_tracking,
    }
    ylabel = plot_args["y"].replace("_", " ").title()
    sns.lineplot(**plot_args, ax=ax, label=ylabel, linewidth=2)

    _beautify_plot(
        ax=ax,
        xlabel=plot_args["x"],
        ylabel=ylabel,
        title="Hyperparameters",
        xlim="tight",
        fontweight="bold",
        facecolor="summary",
    )

    ax.legend()
    _add_last_value_to_legend(ax)
