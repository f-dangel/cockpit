"""Performance Gauge."""

import seaborn as sns

from .utils_instruments import create_basic_plot


def performance_gauge(self, fig, gridspec):
    """[summary]

    Args:
        fig ([type]): [description]
        gridspec ([type]): [description]
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
        "y_scale": "log",
        "cmap": self.cmap,
        "title": "Performance Plot",
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
    sns.lineplot(**plot_args, ax=ax2, label=plot_args["y"], linewidth=2)

    # Train Accuracy
    plot_args = {
        "x": "iteration",
        "y": "valid_accuracy",
        "data": self.epoch_tracking,
    }
    sns.lineplot(**plot_args, ax=ax2, label=plot_args["y"], linewidth=2)
