"""Plotting Part of the Cockpit."""

import glob
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from backboard import instruments
from backboard.instruments import utils_plotting
from deepobs import config


class CockpitPlotter:
    """Cockpit Plotter Class."""

    def __init__(self, logpath):
        """Initialize the cockpit plotter.

        Args:
            logpath (str): Full path to the JSON logfile
        """
        # Split (and store) logpath up to indentify testproblem, data set, etc.
        self.__dict__ = utils_plotting._split_logpath(logpath)

        # Set plotting parameters
        self._set_plotting_params()

    def plot(self, show_plot=True, save_plot=False, savename_append=None, block=False):
        """Plot the cockpit for the current state of the log file.

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
        if not hasattr(self, "fig"):
            self.fig = plt.figure(constrained_layout=False)

        # read in results
        self._read_tracking_results()

        # Plotting
        self.fig.clf()  # clear the cockpit figure to replace it

        # Subplot grid: Currently looks like this.
        # +-----------------------+------------------------+--------------------+
        # | TIC                   | Gradient Tests Gauge | Alpha Gauge          |
        # | Max Ev                | Trace (layerwise)    | Distance (layerwise) |
        # | 1D Histogram          | 2D Histogram         | Grad Norm            |
        # |                         Performance Gauge                           |
        # |                         Hyperparameter Gauge                        |
        # +-----------------------+------------------------+--------------------+
        self.grid_spec = self.fig.add_gridspec(5, 3, wspace=0.15, hspace=0.5)

        # First (upper) Row #
        # TIC
        instruments.tic_gauge(self, self.fig, self.grid_spec[0, 0])

        # Gradient Tests Gauge
        instruments.gradient_tests_gauge(self, self.fig, self.grid_spec[0, 1])
        # Alpha Gauge
        instruments.alpha_gauge(self, self.fig, self.grid_spec[0, 2])

        # Second Row #
        # Max Ev
        instruments.max_ev_gauge(self, self.fig, self.grid_spec[1, 0])
        # Trace (layerwise)
        instruments.trace_gauge(self, self.fig, self.grid_spec[1, 1])
        # Distance (layerwise)
        instruments.distance_gauge(self, self.fig, self.grid_spec[1, 2])

        # Third Row #
        # 1D Histogram
        instruments.histogram_1d_gauge(self, self.fig, self.grid_spec[2, 0])
        # 2D Histogram
        instruments.histogram_2d_gauge(self, self.fig, self.grid_spec[2, 1])
        # Grad Norm
        instruments.grad_norm_gauge(self, self.fig, self.grid_spec[2, 2])

        # Fourth Row #
        instruments.performance_gauge(self, self.fig, self.grid_spec[3, :])

        # Fifth (bottom) Row #
        instruments.hyperparameter_gauge(self, self.fig, self.grid_spec[4, :])

        # Post Process Title, Legend etc.
        self._post_process_plot()

        plt.tight_layout()

        # Show or Save plots
        if show_plot:
            plt.show(block=block)
            plt.pause(0.001)
        if save_plot:
            self._save(savename_append)

    def build_animation(self, duration=200, loop=0):
        """Build an animation from the stored images during training.

        TODO Make this independant of stored images. Instead generate those images
        in hindsight and ideally use fixed axis.

        Args:
            duration (int, optional): Time to display each frame, in milliseconds.
                Defaults to 200.
            loop (int, optional): Number of times the GIF should loop.
                Defaults to 0 which means it will loop forever.
        """
        # Filepaths
        fp_in = self.logpath + "__epoch__*.png"
        fp_out = self.logpath + ".gif"

        # Collect images and create Animation
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=duration,
            loop=loop,
        )

    def _set_plotting_params(self):
        """Set the general plotting options, such as plot size, style, etc."""
        # Settings:
        plt.ion()  # turn on interactive mode, so programm continues while plotting.
        plot_size_default = [30, 15]
        plot_scale = 1.0  # 0.7 works well for the MacBook
        sns.set_style("dark")
        sns.set_context("paper", font_scale=1.0)
        self.save_format = ".png"  # how the plots should be saved
        self.cmap = plt.cm.viridis  # primary color map
        self.cmap2 = plt.cm.cool  # secondary color map
        self.cmap_backup = plt.cm.Wistia  # primary backup color map
        self.cmap2_backup = plt.cm.autumn  # secondary backup color map
        self.color_summary_plots = "#ababba"  # highlight color of summary plots
        self.EMA_alpha = 0.2  # Decay factor of the exponential moving avg.

        # Apply the settings
        mpl.rcParams["figure.figsize"] = [plot_scale * e for e in plot_size_default]

    def _read_tracking_results(self):
        """Read the tracking results from the JSON file into an internal DataFrame."""
        with open(self.logpath) as f:
            data = json.load(f)

        # Read data into a DataFrame
        self.tracking_data = pd.DataFrame.from_dict(data, orient="index")
        # Change data type of index to numeric
        self.tracking_data.index = pd.to_numeric(self.tracking_data.index)
        # Sort by this index
        self.tracking_data = self.tracking_data.sort_index()
        # Rename index to 'iteration' and store it in seperate column
        self.tracking_data = self.tracking_data.rename_axis("iteration").reset_index()

    def _save(self, savename_append=None):
        """Save the (internal) figure to file.

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
