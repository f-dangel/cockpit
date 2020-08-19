"""Plotting Part of the Cockpit"""

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from deepobs import config

from .plotting import instruments, utils_plotting


class CockpitPlotter:
    def __init__(self, logpath):
        """Initialize the cockpit plotter

        Args:
            logpath (str): Full path to the JSON logfile
        """
        # Split (and store) logpath up to indentify testproblem, data set, etc.
        self.__dict__ = utils_plotting._split_logpath(logpath)

        # Set plotting parameters
        self._set_plotting_params()

    def plot(self, show_plot=True, save_plot=False, savename_append=None, block=False):
        """Plot the cockpit for the current state of the log file

        Args:
            show_plot (bool, optional): Whether the plot should be shown on
                screen. Defaults to True.
            save_plot (bool, optional): Whether the plot should be saved to disk.
                Defaults to False.
            savename_append (str, optional): Optional appendix to the savefile
                name. Defaults to None.
            block (bool, optional): Whether the halt the computation after
                blocking or not. Defaults to False.
        """
        # read in results
        self._read_tracking_results()

        # Plotting
        self.fig.clf()  # clear the cockpit figure to replace it

        # Subplot grid: Currently looks like this. Instruments with an index,
        # have a flexible amount of plots (one for each part.)
        # +-----------------------+------------------------+--------------------+
        # | Legend                | Alpha vs. Trace      | Alpha Gauge          |
        # | Max Eigenvalues Gauge | Cond. Number Gauge   |Cond. Number vs. Alpha|
        # | Trace Gauge_0         | Trace Gauge_1        | Trace Gauge_all      |
        # | Distance Gauge_0      | Distance Gauge_1     | Distance Gauge_all   |
        # |                         Performance Gauge                           |
        # |                         Hyperparameter Gauge                        |
        # +-----------------------+------------------------+--------------------+
        self.grid_spec = self.fig.add_gridspec(6, max(self.parts, 2) + 1)

        # First Row #
        # first spot is reserved for the legend which is computed later!
        # Alpha vs Trace
        # instruments.alpha_trace_gauge(self, self.fig, self.grid_spec[0, -2])
        # Alpha Gauge
        # instruments.alpha_gauge(self, self.fig, self.grid_spec[0, -1])

        # Second Row #
        # Max EV
        instruments.max_ev_gauge(self, self.fig, self.grid_spec[1, -3])
        # Cond Number Gauge
        instruments.cond_gauge(self, self.fig, self.grid_spec[1, -2])
        # Cond Number vs Alpha
        instruments.cond_alpha_gauge(self, self.fig, self.grid_spec[1, -1])

        # Third Row
        # per part trace
        for i in range(self.parts):
            instruments.trace_gauge(self, self.fig, self.grid_spec[2, i], part=i)
        # overall trace
        instruments.trace_gauge(self, self.fig, self.grid_spec[2, -1])

        # Fourth Row #
        # per part distance traveled
        for i in range(self.parts):
            instruments.distance_gauge(self, self.fig, self.grid_spec[3, i], part=i)
        # overall distance traveled
        instruments.distance_gauge(self, self.fig, self.grid_spec[3, -1])

        # Fifth Row #
        # mini-batch train loss + train & valid accuracy
        instruments.performance_gauge(self, self.fig, self.grid_spec[4, :])

        # Sixth Row #
        # learning rate
        instruments.hyperparameter_gauge(self, self.fig, self.grid_spec[5, :])

        # compute legend
        self._post_process_plot()

        # Show or Save plots
        if show_plot:
            plt.show(block=block)
            plt.pause(0.001)
        if save_plot:
            self._save(savename_append)

    def _set_plotting_params(self):
        """Set the general plotting options, such as plot size, style, etc."""
        # Settings:
        plt.ion()  # turn on interactive mode, so programm continues while plotting.
        plot_size_default = [20, 10]
        plot_scale = 1.0  # 0.7 works well for the MacBook
        sns.set_style("dark")
        sns.set_context("paper", font_scale=1.0)
        self.save_format = ".svg"  # how the plots should be saved
        self.cmap = plt.cm.viridis  # primary color map
        self.cmap2 = plt.cm.cool  # secondary color map
        self.color_summary_plots = "#ababba"  # highlight color of summary plots
        self.EMA_alpha = 0.2  # Decay factor of the exponential moving avg.

        # Apply the settings
        mpl.rcParams["figure.figsize"] = [plot_scale * e for e in plot_size_default]
        self.fig = plt.figure(constrained_layout=True)

    def _read_tracking_results(self):
        """Read in the tracking results from the JSON file into an internal
        DataFrame.
        """
        with open(self.logpath) as f:
            data = json.load(f)

        self.iter_tracking = pd.DataFrame.from_dict(data["iter_tracking"])
        self.epoch_tracking = pd.DataFrame.from_dict(data["epoch_tracking"])

        # Compute number of layers and parts of the model
        self.layers, self.parts = utils_plotting._compute_layers_parts(self)

        # Process data, e.g. merge layers, etc.
        utils_plotting._process_tracking_results(self)

    def _save(self, savename_append=None):
        """Save the (internal) figure to file

        Args:
            savename_append (str, optional): Optional appendix to the savefile
                name. Defaults to None.
        """
        file_path = (
            os.path.splitext(self.logpath)[0] + self.save_format
            if savename_append is None
            else os.path.splitext(self.logpath)[0] + savename_append + self.save_format
        )

        self.fig.savefig(file_path)

    def _post_process_plot(self):
        """Process the plotting figure, by adding a title, legend, etc."""
        # Set Title
        tp = (
            config.get_data_set_naming()[self.dataset]
            + " "
            + config.get_tp_naming()[self.model]
        )
        self.fig.suptitle(
            "Cockpit for " + self.optimizer + " on " + tp,
            fontsize="xx-large",
            fontweight="bold",
        )

        # # Set Legend
        # ax = self.fig.add_subplot(self.grid_spec[0, 0])
        # ax.legend(han, leg, loc="upper left", ncol=2)
        # ax.set_frame_on(False)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
