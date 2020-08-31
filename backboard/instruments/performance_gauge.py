"""Performance Gauge."""

import seaborn as sns

from .utils_instruments import _add_last_value_to_legend, create_basic_plot


def performance_gauge(self, fig, gridspec):
    """Plotting train/valid accuracy vs. epoch and mini-batch loss vs. iteration.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    # Mini-batch train loss
    plot_args = {
        "x": "iteration",
        "y": "f0",
        "data": self.iter_tracking,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "x_scale": "linear",
        "y_scale": "linear",
        "cmap": self.cmap,
        "ylabel": "Mini-batch losses",
        "title": "Performance Plot",
        "xlim": "tight",
        "ylim": None,
        "fontweight": "bold",
        "facecolor": "summary",
    }
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)

    # Train Accuracy
    plot_args = {
        "x": "iteration",
        "y": "train_accuracy",
        "data": self.epoch_tracking,
    }
    ax2 = ax.twinx()
    sns.lineplot(
        **plot_args, ax=ax2, label=plot_args["y"].title().replace("_", " "), linewidth=2
    )

    # Train Accuracy
    plot_args = {
        "x": "iteration",
        "y": "valid_accuracy",
        "data": self.epoch_tracking,
    }
    sns.lineplot(
        **plot_args, ax=ax2, label=plot_args["y"].title().replace("_", " "), linewidth=2
    )

    # Customization
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Accuracy")
    _add_last_value_to_legend(ax2, percentage=True)
