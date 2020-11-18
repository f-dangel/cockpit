"""Plotting Part of the Cockpit."""

import glob
import json
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from cockpit import instruments
from cockpit.instruments import utils_plotting
from deepobs import config


class CockpitPlotter:
    """Cockpit Plotter Class."""

    def __init__(self, logpath, secondary_screen=False):
        """Initialize the cockpit plotter.

        Args:
            logpath (str): Full path to the JSON logfile.
            secondary_screen (bool): Whether to plot other experimental quantities
                on a secondary screen.
        """
        # Split (and store) logpath up to indentify testproblem, data set, etc.
        self.__dict__ = utils_plotting._split_logpath(logpath)

        self._mpl_default_backend = plt.get_backend()
        self._mpl_no_show_backend = "Agg"

        self._secondary_screen = secondary_screen

        # Set plotting parameters
        self._set_plotting_params()
        self._set_layout_params()

    def _set_layout_params(self):
        """Initialize parameters that define the plot layout."""
        # Individual parts are managed separetely but (for now) they share a layout
        self.inner_num_rows = 5
        self.inner_num_cols = 3
        self.inner_width_ratios = [0.05, 1, 0.05]
        self.inner_height_ratios = [0.0, 1, 1, 1, 0.00]
        self.inner_hspace = 0.6

    def _set_backend(self, show_plot):
        """Use a backend that does (not) show plots."""
        current = plt.get_backend()
        new = self._mpl_default_backend if show_plot else self._mpl_no_show_backend

        if current != new:
            mpl.use(new)

    def plot(
        self,
        show_plot=True,
        save_plot=False,
        savename_append=None,
        block=False,
        show_log_iter=False,
    ):
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
            show_log_iter (bool, optional): Whether the instruments should use
                a log scale for the iterations. Defaults to False.
        """
        self._set_backend(show_plot)

        self.show_log_iter = show_log_iter

        if not hasattr(self, "fig"):
            self.fig = plt.figure("Primary screen", constrained_layout=False)

        # read in results
        self._read_tracking_results()

        # Plotting
        self.fig.clf()  # clear the cockpit figure to replace it

        # Subplot grid: Currently looks like this.
        # +-----------------------+------------------------+--------------------+
        # | STEP SIZE:            | GRADIENTS:           | CURVATURE            |
        # |                       |                      |                      |
        # | Alpha Gauge           | Gradient Tests Gauge | MaxEV                |
        # | Distance              | 1D Histogram         | Trace (layerwise)    |
        # | Grad Norm             | 2D Histogram         | TIC                  |
        # |                                                                     |
        # | Hyperparameter Gauge  |  Performance Gauge                          |
        # +-----------------------+------------------------+--------------------+

        # Build the larger grid (for the three categories, and the two bottom plots)
        outer_widths = [1, 1, 1]
        outer_heights = [3, 1]
        self.grid_spec = self.fig.add_gridspec(
            ncols=3,
            nrows=2,
            width_ratios=outer_widths,
            height_ratios=outer_heights,
            wspace=0.1,
            hspace=0.1,
        )

        self._plot_step(self.grid_spec[0, 0])
        self._plot_gradients(self.grid_spec[0, 1])
        self._plot_curvature(self.grid_spec[0, 2])
        self._plot_hyperparams(self.grid_spec[1, 0])
        self._plot_performance(self.grid_spec[1, 1:])

        # Post Process Title, Legend etc.
        self._post_process_plot()

        if self._secondary_screen:
            self._plot_secondary_screen()

        plt.tight_layout()

        # Show or Save plots
        if show_plot:
            plt.show(block=block)
            plt.pause(0.001)
        if save_plot:
            self._save(savename_append, screen="primary")

            if self._secondary_screen:
                self._save(savename_append, screen="secondary")

    def _plot_step(self, grid_spec):
        """Plot all instruments having to do with step size in the given gridspec.

        Args:
            grid_spec (matplotlib.gridspec): GridSpec where the plot should be placed
        """
        # Use grid_spec with a "dummy plot" to set Group title and color
        self.ax_step = self.fig.add_subplot(grid_spec)
        self.ax_step.set_title("STEP SIZE", fontweight="bold", fontsize="x-large")
        self.ax_step.set_facecolor(self.bg_color_one)
        self.ax_step.set_xticklabels([])
        self.ax_step.set_yticklabels([])

        # Build inner structure of this plotting group
        # We use additional "dummy" gridspecs to position the instruments
        self.gs_step = grid_spec.subgridspec(
            self.inner_num_rows,
            self.inner_num_cols,
            width_ratios=self.inner_width_ratios,
            height_ratios=self.inner_height_ratios,
            hspace=self.inner_hspace,
        )

        instruments.alpha_gauge(self, self.fig, self.gs_step[1, 1])
        instruments.distance_gauge(self, self.fig, self.gs_step[2, 1])
        instruments.grad_norm_gauge(self, self.fig, self.gs_step[3, 1])

    def _plot_gradients(self, grid_spec):
        """Plot all instruments having to do with the gradients in the given gridspec.

        Args:
            grid_spec (matplotlib.gridspec): GridSpec where the plot should be placed
        """
        # Use grid_spec with a "dummy plot" to set Group title and color
        self.ax_gradients = self.fig.add_subplot(grid_spec)
        self.ax_gradients.set_title("GRADIENTS", fontweight="bold", fontsize="x-large")
        self.ax_gradients.set_facecolor(self.bg_color_two)
        self.ax_gradients.set_xticklabels([])
        self.ax_gradients.set_yticklabels([])

        # Build inner structure of this plotting group
        # We use additional "dummy" gridspecs to position the instruments
        self.gs_gradients = grid_spec.subgridspec(
            self.inner_num_rows,
            self.inner_num_cols,
            width_ratios=self.inner_width_ratios,
            height_ratios=self.inner_height_ratios,
            hspace=self.inner_hspace,
        )

        instruments.gradient_tests_gauge(self, self.fig, self.gs_gradients[1, 1])
        instruments.histogram_1d_gauge(self, self.fig, self.gs_gradients[2, 1])
        instruments.histogram_2d_gauge(self, self.fig, self.gs_gradients[3, 1])

    def _plot_curvature(self, grid_spec):
        """Plot all instruments having to do with curvature in the given gridspec.

        Args:
            grid_spec (matplotlib.gridspec): GridSpec where the plot should be placed
        """
        # Use grid_spec with a "dummy plot" to set Group title and color
        self.ax_curvature = self.fig.add_subplot(grid_spec)
        self.ax_curvature.set_title("CURVATURE", fontweight="bold", fontsize="x-large")
        self.ax_curvature.set_facecolor(self.bg_color_three)
        self.ax_curvature.set_xticklabels([])
        self.ax_curvature.set_yticklabels([])

        # Build inner structure of this plotting group
        # We use additional "dummy" gridspecs to position the instruments
        self.gs_curvature = grid_spec.subgridspec(
            self.inner_num_rows,
            self.inner_num_cols,
            width_ratios=self.inner_width_ratios,
            height_ratios=self.inner_height_ratios,
            hspace=self.inner_hspace,
        )

        instruments.max_ev_gauge(self, self.fig, self.gs_curvature[1, 1])
        instruments.trace_gauge(self, self.fig, self.gs_curvature[2, 1])
        instruments.tic_gauge(self, self.fig, self.gs_curvature[3, 1])

    def _plot_hyperparams(self, grid_spec):
        """Plot all instruments showing the hyperparameters.

        Args:
            grid_spec (matplotlib.gridspec): GridSpec where the plot should be placed
        """
        instruments.hyperparameter_gauge(self, self.fig, grid_spec)

    def _plot_performance(self, grid_spec):
        """Plot all instruments having to do with the networks performance.

        Args:
            grid_spec (matplotlib.gridspec): GridSpec where the plot should be placed
        """
        instruments.performance_gauge(self, self.fig, grid_spec)

    def build_animation(
        self,
        duration=200,
        loop=0,
    ):
        """Build an animation from the stored images during training.

        TODO Make this independant of stored images. Instead generate those images
        in hindsight and ideally use fixed axis.

        Args:
            duration (int, optional): Time to display each frame, in milliseconds.
                Defaults to 200.
            loop (int, optional): Number of times the GIF should loop.
                Defaults to 0 which means it will loop forever.
        """
        screens = ["primary"]
        if self._secondary_screen:
            screens.append("secondary")

        for screen in screens:
            fp_out = os.path.splitext(self.logpath)[0] + f"__{screen}.gif"
            self._animate(screen, fp_out, duration, loop)

    def _animate(self, screen, fp_out, duration, loop):
        """Generate animation from paths to images and save."""
        # load frames
        pattern = os.path.splitext(self.logpath)[0] + f"__{screen}__epoch__*.png"
        frame_paths = sorted(glob.glob(pattern))
        frame, *frames = [Image.open(f) for f in frame_paths]

        # Collect images and create Animation
        frame.save(
            fp=fp_out,
            format="GIF",
            append_images=frames,
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
        # Colors #
        self.primary_color = (0.29, 0.45, 0.68, 1.0)  # blue #4a73ad
        self.secondary_color = (0.95, 0.50, 0.20, 1.0)  # orange #f28033
        self.tertiary_color = (0.30, 0.60, 0.40, 1.0)  # green #339966
        # Background colors for the plotting groups
        alpha = 0.75
        self.bg_color_one = self.primary_color[:-1] + (alpha,)
        self.bg_color_two = self.secondary_color[:-1] + (alpha,)
        self.bg_color_three = self.tertiary_color[:-1] + (alpha,)
        self.cmap = plt.cm.viridis  # primary color map
        self.cmap2 = plt.cm.cool  # secondary color map
        self.alpha_cmap = utils_plotting._alpha_cmap(self.primary_color)
        self.bg_color_instruments = (1.0, 1.0, 1.0)
        self.bg_color_instruments2 = "#ababba"  # highlight color of summary plots

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

    def _save(self, savename_append=None, screen="primary"):
        """Save the (internal) figure to file.

        Args:
            savename_append (str, optional): Optional appendix to the savefile
                name. Defaults to None.
            screen (str): String that specifies screen figure should be saved.
                Possible options are ``'primary'`` and ``'secondary'``.
        """
        if savename_append is None:
            savename_append = ""

        file_path = (
            os.path.splitext(self.logpath)[0]
            + f"__{screen}"
            + savename_append
            + self.save_format
        )

        if screen == "primary":
            self.fig.savefig(file_path)
        elif screen == "secondary":
            self.secondary_fig.savefig(file_path)
        else:
            raise ValueError(f"screen must be 'primary' or 'secondar'y. Got {screen}")

    def _post_process_plot(self):
        """Process the plotting figure, by adding a title, legend, etc."""
        # Set Title
        try:
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
        except KeyError:
            self.fig.suptitle(
                "Cockpit for " + self.optimizer, fontsize="xx-large", fontweight="bold"
            )

        # # Set Legend
        # ax = self.fig.add_subplot(self.grid_spec[0, 0])
        # ax.legend(han, leg, loc="upper left", ncol=2)
        # ax.set_frame_on(False)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    def _plot_secondary_screen(self):
        """Plot a second figure with experimental quantities."""
        if not hasattr(self, "secondary_fig"):
            self.secondary_fig = plt.figure(
                "Secondary screen", constrained_layout=False
            )

        self.secondary_fig.clf()

        secondary_outer_widths = [1]
        secondary_outer_heights = [1, 3]

        self.secondary_grid_spec = self.secondary_fig.add_gridspec(
            ncols=1,
            nrows=2,
            width_ratios=secondary_outer_widths,
            height_ratios=secondary_outer_heights,
            wspace=0.1,
            hspace=0.1,
        )

        self.__set_ax_auxiliary(self.secondary_grid_spec[0, 0])
        self._plot_auxiliary(self.secondary_grid_spec[0, 0])

        self.__set_ax_layerwise(self.secondary_grid_spec[1, 0])
        self._plot_layerwise(self.secondary_grid_spec[1, 0])

    def __set_ax_auxiliary(self, grid_spec):
        """Use grid_spec with a "dummy plot" to set Group title and color."""
        self.ax_auxiliary = self.secondary_fig.add_subplot(grid_spec)
        self.ax_auxiliary.set_title("AUXILIARY", fontweight="bold", fontsize="x-large")
        self.ax_auxiliary.set_facecolor(self.bg_color_one)
        self.ax_auxiliary.set_xticklabels([])
        self.ax_auxiliary.set_yticklabels([])

    def _plot_auxiliary(self, grid_spec):
        """Plot auxiliary quantities to the secondary screen."""

        # Build inner structure of this plotting group
        # We use additional "dummy" gridspecs to position the instruments
        self.gs_auxiliary = grid_spec.subgridspec(
            3,
            7,
            width_ratios=[0.05, 1, 0.05, 1, 0.05, 1, 0.05],
            height_ratios=[0.0, 1, 0.0],
            hspace=self.inner_hspace,
        )

        # plot mean GS NR
        instruments.mean_gsnr_gauge(self, self.secondary_fig, self.gs_auxiliary[1, 1])
        instruments.cabs_gauge(self, self.secondary_fig, self.gs_auxiliary[1, 3])
        instruments.early_stopping_gauge(
            self, self.secondary_fig, self.gs_auxiliary[1, 5]
        )

    def __set_ax_layerwise(self, grid_spec):
        """Use grid_spec with a "dummy plot" to set Group title and color."""
        self.ax_layerwise = self.secondary_fig.add_subplot(grid_spec)
        self.ax_layerwise.set_title("LAYERWISE", fontweight="bold", fontsize="x-large")
        self.ax_layerwise.set_facecolor(self.bg_color_two)
        self.ax_layerwise.set_xticklabels([])
        self.ax_layerwise.set_yticklabels([])

    def _plot_layerwise(self, grid_spec, fig=None):
        """Plot layerwise 2d histograms to the secondary screen."""

        # Build inner structure
        try:
            param_groups = int(
                self.tracking_data[["param_groups"]].dropna().tail(1).to_numpy()
            )
        except KeyError:
            warnings.warn("Cannot create layerwise plots (missing 'param_groups')")
            return

        def get_layout(num_plots, min_rows=2, min_cols=2):
            """Step-wise increase rows and columns until they can fit all plots."""
            dims = [min_rows, min_cols]

            increase_next = 0
            while dims[0] * dims[1] < num_plots:
                dims[increase_next] = dims[increase_next] + 1
                increase_next = (increase_next + 1) % 2

            return dims

        num_rows, num_cols = get_layout(param_groups)

        self.gs_layerwise = grid_spec.subgridspec(
            2 * num_rows + 1,
            2 * num_cols + 1,
            width_ratios=num_cols * [0.05, 1] + [0.05],
            height_ratios=num_rows * [0.0, 1] + [0.0],
            hspace=self.inner_hspace,
        )

        def to_grid(idx):
            """Map one-dimension index to coordinates in 2d layout.

            Need to take into account the padding around actual plots.
            """
            assert 0 <= idx < param_groups
            x_unpadded, y_unpadded = divmod(idx, num_cols)
            return 2 * x_unpadded + 1, 2 * y_unpadded + 1

        if fig is None:
            fig = self.secondary_fig

        for idx in range(param_groups):
            x, y = to_grid(idx)
            instruments.histogram_2d_gauge(self, fig, self.gs_layerwise[x, y], idx=idx)
