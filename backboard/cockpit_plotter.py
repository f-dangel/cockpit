"""Plotting Part of the Cockpit"""
import json
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from backboard.utils.cockpit_utils import _root_sum_of_squares
from deepobs import config


class CockpitPlotter:
    """Plotting class for the Deep Learning Cockpit."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.optimizer = self.log_path.split("/")[-3]
        self.testproblem = self.log_path.split("/")[-4]
        self.tp_dataset = self.testproblem.split("_")[0]
        self.tp_model = self.testproblem.split("_")[1]

        self._set_plotting_parameters()

        # Will be set the first time we plot
        self.iter_per_plot = 0

    def plot(self, show=True, save=False, save_append=None):
        """Shows a cockpit plot using the cockpit plotter and the current log file."""

        self._read_tracking_results()

        # The first time we plot, compute how many iterations are between two plots.
        if self.iter_per_plot == 0:
            self.iter_per_plot = len(self.iter_tracking["f0"])

        plt.ion()

        if hasattr(self, "fig"):
            self.fig.clf()
        else:
            self.fig = plt.figure(constrained_layout=True)

        self.grid_spec = self.fig.add_gridspec(6, self.parts + 1)

        # # f gauge
        # self._plot_f_rel(self.grid_spec[0, -1])

        # alpha vs trace gauge
        self._plot_alpha_trace(self.grid_spec[0, -2])

        # alpha gauge
        self._plot_alpha(self.grid_spec[0, -1])

        # # df gauge
        # for i in range(self.n_layers):
        #     self._plot_df_rel(self.grid_spec[1, i], i)
        # self._plot_df_rel(self.grid_spec[1, -1])\

        # min vs max eigenvalue
        self._plot_min_max_ev(self.grid_spec[1, 0])

        # alpha vs cond. number
        self._plot_alpha_cond(self.grid_spec[1, 1])

        # cond. number vs. iteration
        self._plot_cond(self.grid_spec[1, 2])

        # trace gauge
        for i in range(self.parts):
            self._plot_trace(self.grid_spec[2, i], i)
        self._plot_trace(self.grid_spec[2, -1])

        # # grad norm gauge
        # for i in range(self.n_layers):
        #     self._plot_grad_norm(self.grid_spec[2, i], i)
        # self._plot_grad_norm(self.grid_spec[2, -1])

        # d2init gauge
        for i in range(self.parts):
            self._plot_dist(self.grid_spec[3, i], i)
        self._plot_dist(self.grid_spec[3, -1])

        # Performance plot
        han, leg = self._plot_perf(self.grid_spec[4, :])

        # Hyperparameter Plot
        self._plot_hyperparams(self.grid_spec[5, :])

        # Set Title
        tp = (
            config.get_data_set_naming()[self.tp_dataset]
            + " "
            + config.get_tp_naming()[self.tp_model]
        )
        self.fig.suptitle(
            "Cockpit for " + self.optimizer + " on " + tp,
            fontsize="xx-large",
            fontweight="bold",
        )

        ax = self.fig.add_subplot(self.grid_spec[0, 0])
        ax.legend(han, leg, loc="upper left", ncol=2)
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if show:
            plt.show()
            print("Cockpit-Plot shown...")
            plt.pause(0.01)
        else:
            plt.close(self.fig)
            print("Cockpit-Plot drawn...")
            plt.pause(0.01)

        if save:
            file_path = (
                self.log_path + ".svg"
                if save_append is None
                else self.log_path + save_append + ".svg"
            )
            self.fig.savefig(file_path)
            print("Cockpit-Plot saved...")

    def save_plot(self):
        """Saves the last cockpit plot to image file."""
        self.fig.savefig(self.log_path + ".svg")
        print("Cockpit-Plot saved...")

    def _set_plotting_parameters(self):
        """Add options like plot_limits, etc."""
        self.plot_size_default = [20, 10]
        self.plot_scale = 1.0  # 0.7 for MacBook

        sns.set_style("dark")
        matplotlib.rcParams["figure.figsize"] = [
            self.plot_scale * e for e in self.plot_size_default
        ]
        sns.set_context("paper", font_scale=1.0)

        self.cmap = plt.cm.viridis
        self.cmap2 = plt.cm.cool
        self.color_summary_plots = "#ababba"

        self.EMA_span = 0.2

    def _read_tracking_results(self):
        with open(self.log_path + ".json") as f:
            data = json.load(f)

        self.iter_tracking = pd.DataFrame.from_dict(data["iter_tracking"])

        self.epoch_tracking = pd.DataFrame.from_dict(data["epoch_tracking"])

        self._process_tracking_results()

    def _process_tracking_results(self):
        """Process the tracking results."""
        # Compute the number of "layers" of the net, and into how many parts
        # we want to split it
        self.n_layers, self.parts = self._number_of_parts()

        # some util variables for the splitting/aggregation
        layers_per_part = self.n_layers // self.parts
        rest_ = self.n_layers % self.parts
        # splits = [layers_per_part + (1 if i < rest_ else 0)
        # for i in range(self.parts)]

        # split part-wise
        # Create new columns for each part
        for (columnName, columnData) in self.iter_tracking.items():
            # We only need to handle data that is non-scalar
            if isinstance(columnData[0], list):
                aggregate = self._get_aggregate_function(columnName)
                # Create new parts
                for p in range(self.parts):
                    start = p * layers_per_part + min(p, rest_)
                    end = (p + 1) * layers_per_part + min(p + 1, rest_)
                    self.iter_tracking[columnName + "_part_" + str(p)] = [
                        aggregate(x[start:end])
                        for x in self.iter_tracking[columnName].tolist()
                    ]
                # Overall average
                self.iter_tracking[columnName] = [
                    aggregate(x[:]) for x in self.iter_tracking[columnName].tolist()
                ]

        # Compute avg_ev & avg_cond
        # for that we need the number of parameters, which for now, we hardcode
        num_params = {
            "mnist_logreg": 7850,
            "cifar10_3c3d": 895210,
            "fmnist_2c2d": 3274634,
        }
        if self.testproblem in num_params:
            self.iter_tracking["avg_ev"] = (
                self.iter_tracking["trace"] / num_params[self.testproblem]
            )
            self.iter_tracking["avg_cond"] = (
                self.iter_tracking["max_ev"] / self.iter_tracking["avg_ev"]
            )
        else:
            warnings.warn(
                "Warning: Unknown testproblem "
                + self.testproblem
                + ", couldn't compute the average eigenvalue",
                stacklevel=1,
            )

    def _get_aggregate_function(self, quantity):
        """Get the corresponding aggregation function for a given quantity.

        Args:
            quantity ([type]): [description]
        """
        if quantity in [
            "df0",
            "df1",
            "var_df0",
            "var_df1",
            "df_var0",
            "df_var1",
            "trace",
        ]:
            return sum
        elif quantity in ["grad_norms", "d2init", "dtravel"]:
            return _root_sum_of_squares
        else:
            warnings.warn(
                "Warning: Don't know how to aggregate " + quantity, stacklevel=2,
            )

    def _number_of_parts(self):
        """Compute the number of parts we want to plot."""
        # check number of layers using df0 variable
        n_layers = len(self.iter_tracking["df0"][0])

        # limit to four parts
        parts = min(n_layers, 4)

        # or use hard-coded setting
        tp_model_parts = {
            "logreg": 2,
            "3c3d": 2,
            "2c2d": 4,
        }
        if self.tp_model in tp_model_parts:
            parts = tp_model_parts[self.tp_model]
        return n_layers, parts

    def _plot_hyperparams(self, gridspec, fig=None):
        """Creates a plot of hyperparameters

        Args:
            gridspec ([gridspec]): Gridspec to plot into
        """
        # Plot Settings
        x_quan = "iteration"
        y_quan = "learning_rate"
        x_scale = "linear"
        y_scale = "linear"
        title = "Hyperparameters"

        # Plotting
        if fig is None:
            ax = self.fig.add_subplot(gridspec)
        else:
            ax = fig.add_subplot(gridspec)

        sns.lineplot(
            x=x_quan,
            y=y_quan,
            # hue="epoch",
            # palette=self.cmap,
            # edgecolor=None,
            # s=10,
            data=self.epoch_tracking,
            ax=ax,
            label=y_quan,
            color="black",
            linewidth=0.8,
        )

        # Customize Plot
        self._customize_epoch_plot(
            ax,
            title,
            x_scale=x_scale,
            y_scale=y_scale,
            fontweight="bold",
            facecolor=self.color_summary_plots,
        )

    def _plot_perf(self, gridspec, fig=None):
        """Create a performance plot."""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "f0"
        y_quan2 = "train_accuracy"
        y_quan3 = "valid_accuracy"
        x_scale = "linear"
        y_scale = "log"
        y_scale2 = "linear"
        title = "Performance Plot"

        # Plotting
        if fig is None:
            ax = self.fig.add_subplot(gridspec)
        else:
            ax = fig.add_subplot(gridspec)
        self.iter_tracking["EMA_" + y_quan] = (
            self.iter_tracking[y_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        sns.scatterplot(
            x=x_quan,
            y=y_quan,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        ax2 = ax.twinx()

        sns.lineplot(
            x=x_quan,
            y=y_quan2,
            # hue="epoch",
            # palette=self.cmap,
            # edgecolor=None,
            # s=10,
            data=self.epoch_tracking,
            ax=ax2,
            label=y_quan2,
            linewidth=2,
        )

        sns.lineplot(
            x=x_quan,
            y=y_quan3,
            # hue="epoch",
            # palette=self.cmap,
            # edgecolor=None,
            # s=10,
            data=self.epoch_tracking,
            ax=ax2,
            label=y_quan3,
            linewidth=2,
        )

        # Customize Plot
        self._customize_plot(
            ax,
            x_quan,
            y_quan,
            x_scale,
            y_scale,
            title,
            fontweight="bold",
            facecolor=self.color_summary_plots,
            extend_factors=[0.0, 0.0],
        )

        # Customize further
        # Set Accuracies limites to 0 and 1
        ax2.set_ylim([0, 1])
        # Set Scales
        ax2.set_yscale(y_scale2)
        # Fix Legend
        lines2, labels2 = ax2.get_legend_handles_labels()
        plot_labels = []
        for line, label in zip(lines2, labels2):
            plot_labels.append("{0}: ({1:.2%})".format(label, line.get_ydata()[-1]))
        ax2.get_legend().remove()
        ax.legend(lines2, plot_labels)

        ax.set_ylabel("Minibatch Train Loss")
        handles, labels = ax.get_legend_handles_labels()

        labels[0] = "Iteration"
        labels[5] = "Running Average"

        return handles, labels

    def _plot_f_rel(self, gridspec):
        """Plot the function value relationship plot"""
        # Plot Settings
        x_quan = "diff_f"
        y_quan = "cert_sign_f"
        x_scale = "linear"
        y_scale = "linear"
        title = "f gauge for the Net"
        ylim = None

        # Compute derived quantities
        if x_quan == "rel_diff_f" or y_quan == "rel_diff_f":
            self.iter_tracking["rel_diff_f"] = (
                self.iter_tracking["f1"] - self.iter_tracking["f0"]
            ) / self.iter_tracking["f0"]
        if x_quan == "diff_f" or y_quan == "diff_f":
            self.iter_tracking["diff_f"] = (
                self.iter_tracking["f1"] - self.iter_tracking["f0"]
            )
        if x_quan == "avg_var_f" or y_quan == "avg_var_f":
            self.iter_tracking["avg_var_f"] = (
                0.5 * self.iter_tracking["var_f1"] + 0.5 * self.iter_tracking["var_f0"]
            )
        if x_quan == "cert_sign_f" or y_quan == "cert_sign_f":
            self.iter_tracking["cert_sign_f"] = self.iter_tracking.apply(
                lambda row: np.abs(
                    stats.norm.cdf(
                        0,
                        loc=(row["f1"] - row["f0"]),
                        scale=(np.sqrt(row["var_f0"] + row["var_f1"])),
                    )
                    - 0.5
                )
                * 2.0,
                axis=1,
            )
            ylim = [0, 1]
        self.iter_tracking["EMA_" + x_quan] = (
            self.iter_tracking[x_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )
        self.iter_tracking["EMA_" + y_quan] = (
            self.iter_tracking[y_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        ax = self.fig.add_subplot(gridspec)
        sns.scatterplot(
            x=x_quan,
            y=y_quan,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x="EMA_" + x_quan,
            y="EMA_" + y_quan,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        # Customize Plot
        self._customize_plot(
            ax,
            x_quan,
            y_quan,
            x_scale,
            y_scale,
            title,
            fontweight="bold",
            facecolor=self.color_summary_plots,
            center=[True, False],
            ylim=ylim,
        )

    def _plot_alpha(self, gridspec, fig=None):  # noqa: C901
        """Plot the local step length"""
        # Plot Settings
        title = "Alpha gauge"
        fontweight = "bold"
        facecolor = self.color_summary_plots
        x_scale = "linear"
        y_scale = "linear"
        xlim = [-2, 2]
        ylim = [0, 2.25]

        # Plotting
        if fig is None:
            ax = self.fig.add_subplot(gridspec)
        else:
            ax = fig.add_subplot(gridspec)

        # Plot unit parabola
        x = np.linspace(xlim[0], xlim[1], 100)
        y = x ** 2
        ax.plot(x, y, linewidth=2)

        # Alpha Histogram
        ax2 = ax.twinx()
        # All alphas
        try:
            sns.distplot(
                self.iter_tracking["alpha"],
                ax=ax2,
                # norm_hist=True,
                fit=stats.norm,
                kde=False,
                color="gray",
                fit_kws={"color": "gray"},
                hist_kws={"linewidth": 0, "alpha": 0.25},
                label="all",
            )
        except ValueError:
            print("Alphas included NaN and could therefore not be plotted.")

        # Just from last plot
        try:
            sns.distplot(
                self.iter_tracking["alpha"][-self.iter_per_plot :].fillna(1000),
                ax=ax2,
                # norm_hist=True,
                fit=stats.norm,
                kde=False,
                color=sns.color_palette()[1],
                fit_kws={"color": sns.color_palette()[1]},
                hist_kws={"linewidth": 0, "alpha": 0.65},
                label="since last plot",
            )
        except ValueError:
            print("Alphas included NaN and could therefore not be plotted.")

        # Customize Plot
        ax.set_title(title, fontweight=fontweight)
        # Zone Lines
        ax.axvline(0, ls="-", color="white", linewidth=1.5, zorder=0)
        ax.axvline(-1, ls="-", color="white", linewidth=1.5, zorder=0)
        ax.axvline(1, ls="-", color="white", linewidth=1.5, zorder=0)
        ax.axhline(0, ls="-", color="white", linewidth=1.5, zorder=0)
        ax.axhline(1, ls="-", color="gray", linewidth=0.5, zorder=0)
        if facecolor is not None:
            ax.set_facecolor(self.color_summary_plots)
        ax.set_xlabel(r"Local step length $\alpha$")
        ax.set_ylabel("Stand. loss")
        ax2.set_ylabel(r"$\alpha$ density")
        ax.set_yscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        # Add indicator for outliers
        if max(self.iter_tracking["alpha"][-self.iter_per_plot :]) > xlim[1]:
            ax.annotate(
                "",
                xy=(1.8, 0.3),
                xytext=(1.7, 0.3),
                size=20,
                arrowprops=dict(color=sns.color_palette()[1]),
            )
        elif max(self.iter_tracking["alpha"]) > xlim[1]:
            ax.annotate(
                "",
                xy=(1.8, 0.3),
                xytext=(1.7, 0.3),
                size=20,
                arrowprops=dict(color="gray"),
            )
        if min(self.iter_tracking["alpha"][-self.iter_per_plot :]) < xlim[0]:
            ax.annotate(
                "",
                xy=(-1.8, 0.3),
                xytext=(-1.7, 0.3),
                size=20,
                arrowprops=dict(color=sns.color_palette()[1]),
            )
        elif min(self.iter_tracking["alpha"]) < xlim[0]:
            ax.annotate(
                "",
                xy=(-1.8, 0.3),
                xytext=(-1.7, 0.3),
                size=20,
                arrowprops=dict(color="gray"),
            )

        # Legend
        # Get the fitted parameters used by sns
        try:
            (mu_all, _) = stats.norm.fit(self.iter_tracking["alpha"])
        except RuntimeError:
            mu_all = None

        try:
            (mu_last, _) = stats.norm.fit(
                self.iter_tracking["alpha"][-self.iter_per_plot :]
            )
        except RuntimeError:
            mu_last = None
        lines2, labels2 = ax2.get_legend_handles_labels()
        try:
            ax2.legend(
                [
                    "{0} ($\mu=${1:.2f})".format(labels2[0], mu_all),  # noqa: W605
                    "{0} ($\mu=${1:.2f})".format(labels2[1], mu_last),  # noqa: W605
                ]
            )
        except TypeError:
            pass

    def _plot_alpha_trace(self, gridspec):
        """Plot the local step size vs the trace over time."""
        # Plot Settings
        x_quan = "trace"
        y_quan = "alpha"
        x_scale = "linear"
        y_scale = "linear"
        title = "Alpha vs Trace"
        fontweight = "bold"
        facecolor = self.color_summary_plots
        ylim = [-2, 2]

        self.iter_tracking["EMA_" + x_quan] = (
            self.iter_tracking[x_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )
        self.iter_tracking["EMA_" + y_quan] = (
            self.iter_tracking[y_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        ax = self.fig.add_subplot(gridspec)
        sns.scatterplot(
            x=x_quan,
            y=y_quan,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x="EMA_" + x_quan,
            y="EMA_" + y_quan,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        self._customize_plot(
            ax,
            x_quan,
            y_quan,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            center=[False, True],
            ylim=ylim,
        )

    def _plot_df_rel(self, gridspec, layer="all"):
        """Plot the derivative value relationship plot"""
        # Plot Settings
        x_quan = "ratio_df"
        y_quan = "cert_sign_df"
        x_scale = "linear"
        y_scale = "linear"
        n = "" if isinstance(layer, str) else "_layer_" + str(layer)
        ylim = None

        # Compute derived quantities
        if x_quan == "ratio_df" or y_quan == "ratio_df":
            self.iter_tracking["ratio_df" + n] = (
                -self.iter_tracking["df1" + n] / self.iter_tracking["df0" + n]
            )
        if x_quan == "avg_df_var" or y_quan == "avg_df_var":
            self.iter_tracking["avg_df_var" + n] = (
                0.5 * self.iter_tracking["df_var_1" + n]
                + 0.5 * self.iter_tracking["df_var_0" + n]
            )
        if x_quan == "cert_sign_df" or y_quan == "cert_sign_df":
            self.iter_tracking["cert_sign_df" + n] = self.iter_tracking.apply(
                lambda row: np.abs(
                    stats.norm.cdf(
                        0,
                        loc=(row["df1" + n] - row["df0" + n]),
                        scale=(np.sqrt(row["var_df_0" + n] + row["var_df_1" + n])),
                    )
                    - 0.5
                )
                * 2.0,
                axis=1,
            )
            ylim = [0, 1]
        self.iter_tracking["EMA_" + x_quan + n] = (
            self.iter_tracking[x_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
        )
        self.iter_tracking["EMA_" + y_quan + n] = (
            self.iter_tracking[y_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        ax = self.fig.add_subplot(gridspec)
        sns.scatterplot(
            x=x_quan + n,
            y=y_quan + n,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x="EMA_" + x_quan + n,
            y="EMA_" + y_quan + n,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        # Customize Plot
        if isinstance(layer, str):
            title = "df gauge for the Net"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "df gauge for Part " + str(layer)
            fontweight = "normal"
            facecolor = None
        self._customize_plot(
            ax,
            x_quan + n,
            y_quan + n,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            center=[True, False],
            ylim=ylim,
        )

    def _plot_alpha_iteration(self, gridspec, fig=None):
        """Plot the trace"""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "alpha"
        x_scale = "linear"
        y_scale = "linear"
        title = "Alpha Plot"
        fontweight = "bold"
        facecolor = self.color_summary_plots

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan] = (
            self.iter_tracking[y_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        if fig is None:
            ax = self.fig.add_subplot(gridspec)
        else:
            ax = fig.add_subplot(gridspec)

        sns.scatterplot(
            x=x_quan,
            y=y_quan,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        self._customize_plot(
            ax,
            x_quan,
            y_quan,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            ylim=[-1, 1],
            center=[False, True],
            extend_factors=[0.0, 0.05],
        )

        # Set y label with average
        avg_alpha = np.mean(self.iter_tracking["alpha"])
        std_alpha = np.std(self.iter_tracking["alpha"])

        ax.set_ylabel(
            r"alpha ($\mu$={0:.2f}, $\sigma$={1:.2f})".format(avg_alpha, std_alpha)
        )

    def _plot_trace(self, gridspec, layer="all", fig=None):
        """Plot the trace"""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "trace"
        x_scale = "linear"
        y_scale = "log"
        n = "" if isinstance(layer, str) else "_part_" + str(layer)

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan + n] = (
            self.iter_tracking[y_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        if fig is None:
            ax = self.fig.add_subplot(gridspec)
        else:
            ax = fig.add_subplot(gridspec)

        sns.scatterplot(
            x=x_quan,
            y=y_quan + n,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan + n,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        # Customize Plot
        if isinstance(layer, str):
            title = "Trace for the Net"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "Trace for Part " + str(layer)
            fontweight = "normal"
            facecolor = None
        self._customize_plot(
            ax,
            x_quan,
            y_quan + n,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            extend_factors=[0.0, 0.0],
        )

    def _plot_trace_cond(self, gridspec, layer="all", fig=None):
        """Plot the trace"""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "trace"
        y_quan2 = "avg_cond"
        x_scale = "linear"
        y_scale = "log"
        n = "" if isinstance(layer, str) else "_part_" + str(layer)

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan + n] = (
            self.iter_tracking[y_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
        )
        self.iter_tracking["EMA_" + y_quan2] = (
            self.iter_tracking[y_quan2].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        if fig is None:
            ax = self.fig.add_subplot(gridspec)
        else:
            ax = fig.add_subplot(gridspec)

        sns.scatterplot(
            x=x_quan,
            y=y_quan + n,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan + n,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        ax2 = ax.twinx()

        sns.scatterplot(
            x=x_quan,
            y=y_quan2,
            hue="iteration",
            palette=self.cmap,
            marker="+",
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax2,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan2,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax2,
        )

        # Customize Plot
        if isinstance(layer, str):
            title = "Trace and (avg.) Cond. Number of the Net"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "Trace and (avg.) Cond. Number of Part " + str(layer)
            fontweight = "normal"
            facecolor = None
        self._customize_plot(
            ax,
            x_quan,
            y_quan + n,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            extend_factors=[0.0, 0.0],
        )
        ax2.get_legend().remove()
        ax2.set_ylim(bottom=min(self.iter_tracking[y_quan2]))
        ax2.set_ylim(top=max(self.iter_tracking[y_quan2]))
        ax2.set_ylabel(y_quan2.capitalize().replace("_", " "))

    def _plot_grad_norm(self, gridspec, layer="all", fig=None):
        """Plot the gradient norm gauge"""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "grad_norms"
        x_scale = "linear"
        y_scale = "linear"
        n = "" if isinstance(layer, str) else "_part_" + str(layer)

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan + n] = (
            self.iter_tracking[y_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        if fig is None:
            ax = self.fig.add_subplot(gridspec)
        else:
            ax = fig.add_subplot(gridspec)

        sns.scatterplot(
            x=x_quan,
            y=y_quan + n,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan + n,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        # Customize Plot
        if isinstance(layer, str):
            title = "Grad Norm Gauge for the Net"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "Grad Norm Gauge for Part " + str(layer)
            fontweight = "normal"
            facecolor = None
        self._customize_plot(
            ax,
            x_quan,
            y_quan + n,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            extend_factors=[0.0, 0.05],
        )

    def _plot_alpha_cond(self, gridspec):
        """Plot the local step length alpha vs the condition number.

        NOTE: The condition number is computed as max_ev/avg_ev,
        not max_ev/min_ev, since the min_ev can become negative due to
        numerical noise. We call this the average condition number avg_cond.
        """
        """Plot the min vs the max eigenvalue of the Hessian over time."""
        # Plot Settings
        x_quan = "avg_cond"
        y_quan = "alpha"
        x_scale = "log"
        y_scale = "linear"
        title = "(Average) Condition Number vs. Alpha"
        fontweight = "bold"
        facecolor = self.color_summary_plots

        # Compute derived quantities
        self.iter_tracking["EMA_" + x_quan] = (
            self.iter_tracking[x_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )
        self.iter_tracking["EMA_" + y_quan] = (
            self.iter_tracking[y_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        ax = self.fig.add_subplot(gridspec)
        sns.scatterplot(
            x=x_quan,
            y=y_quan,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x="EMA_" + x_quan,
            y="EMA_" + y_quan,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        self._customize_plot(
            ax,
            x_quan,
            y_quan,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            extend_factors=[0.05, 0.05],
        )

    def _plot_cond(self, gridspec):
        """Plot the condition number vs iteration.

        NOTE: The condition number is computed as max_ev/avg_ev,
        not max_ev/min_ev, since the min_ev can become negative due to
        numerical noise. We call this the average condition number avg_cond.
        """
        # Plot Settings
        x_quan = "iteration"
        y_quan = "avg_cond"
        x_scale = "linear"
        y_scale = "linear"
        title = "(Average) Condition Number"
        fontweight = "bold"
        facecolor = self.color_summary_plots

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan] = (
            self.iter_tracking[y_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        ax = self.fig.add_subplot(gridspec)
        sns.scatterplot(
            x=x_quan,
            y=y_quan,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        self._customize_plot(
            ax,
            x_quan,
            y_quan,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            extend_factors=[0.0, 0.05],
        )

    def _plot_min_max_ev(self, gridspec):
        """Plot the min vs the max eigenvalue of the Hessian over time."""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "max_ev"
        x_scale = "linear"
        y_scale = "log"
        title = "Max Eigenvalue"
        fontweight = "bold"
        facecolor = self.color_summary_plots

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan] = (
            self.iter_tracking[y_quan].ewm(alpha=self.EMA_span, adjust=False).mean()
        )

        # Plotting
        ax = self.fig.add_subplot(gridspec)
        sns.scatterplot(
            x=x_quan,
            y=y_quan,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        self._customize_plot(
            ax,
            x_quan,
            y_quan,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            extend_factors=[0.0, 0.05],
        )

    def _plot_dist(self, gridspec, layer="all"):
        """Plot the distance."""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "d2init"
        y_quan2 = "dtravel"
        x_scale = "linear"
        y_scale = "linear"
        n = "" if isinstance(layer, str) else "_part_" + str(layer)

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan + n] = (
            self.iter_tracking[y_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
        )
        self.iter_tracking["EMA_" + y_quan2 + n] = (
            self.iter_tracking[y_quan2 + n]
            .ewm(alpha=self.EMA_span, adjust=False)
            .mean()
        )

        # Plotting
        ax = self.fig.add_subplot(gridspec)
        sns.scatterplot(
            x=x_quan,
            y=y_quan + n,
            hue="iteration",
            palette=self.cmap,
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan + n,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax,
        )

        ax2 = ax.twinx()

        sns.scatterplot(
            x=x_quan,
            y=y_quan2 + n,
            hue="iteration",
            palette=self.cmap,
            marker="+",
            edgecolor=None,
            s=10,
            data=self.iter_tracking,
            ax=ax2,
        )
        sns.scatterplot(
            x=x_quan,
            y="EMA_" + y_quan2 + n,
            hue="iteration",
            palette=self.cmap2,
            marker=",",
            edgecolor=None,
            s=1,
            data=self.iter_tracking,
            ax=ax2,
        )

        # Customize Plot
        if isinstance(layer, str):
            title = "Distance Gauge for the Net"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "Distance Gauge for Part " + str(layer)
            fontweight = "normal"
            facecolor = None
        self._customize_plot(
            ax,
            x_quan,
            y_quan + n,
            x_scale,
            y_scale,
            title,
            fontweight,
            facecolor,
            extend_factors=[0.0, 0.05],
        )
        ax2.get_legend().remove()
        ax2.set_ylim(bottom=0)
        ax2.set_ylim(top=max(self.iter_tracking[y_quan2 + n]))
        ax2.set_ylabel(y_quan2.capitalize().replace("_", " "))

    def _customize_plot(
        self,
        ax,
        x_quan,
        y_quan,
        x_scale,
        y_scale,
        title,
        fontweight="normal",
        facecolor=None,
        extend_factors=None,
        center=None,
        xlim=None,
        ylim=None,
    ):
        """Customize Plot with labels, titels etc,"""
        ax.set_title(title, fontweight=fontweight)
        # Zero lines
        ax.axvline(0, ls="-", color="white", linewidth=1.5, zorder=0)
        ax.axhline(0, ls="-", color="white", linewidth=1.5, zorder=0)
        if facecolor is not None:
            ax.set_facecolor(self.color_summary_plots)
        ax.set_xlabel(x_quan.capitalize().replace("_", " "))
        ax.set_ylabel(y_quan.capitalize().replace("_", " "))
        ax.get_legend().remove()
        ax.set_yscale(x_scale)
        ax.set_yscale(y_scale)

        if extend_factors is None:
            extend_factors = [0.05, 0.05]
        if center is None:
            center = [False, False]

        xlim_min = min(self.iter_tracking[x_quan]) - extend_factors[0] * (
            max(self.iter_tracking[x_quan]) - min(self.iter_tracking[x_quan])
        )
        xlim_max = max(self.iter_tracking[x_quan]) + extend_factors[0] * (
            max(self.iter_tracking[x_quan]) - min(self.iter_tracking[x_quan])
        )

        if center[0]:
            xlim_max = abs(max([xlim_min, xlim_max], key=abs))
            xlim_min = -xlim_max
        ylim_min = min(self.iter_tracking[y_quan]) - extend_factors[1] * (
            max(self.iter_tracking[y_quan]) - min(self.iter_tracking[y_quan])
        )
        ylim_max = max(self.iter_tracking[y_quan]) + extend_factors[1] * (
            max(self.iter_tracking[y_quan]) - min(self.iter_tracking[y_quan])
        )
        if center[1]:
            ylim_max = abs(max([ylim_min, ylim_max], key=abs))
            ylim_min = -ylim_max
        if xlim is None:
            ax.set_xlim([xlim_min, xlim_max])
        else:
            ax.set_xlim(xlim)

        if ylim is None:
            ax.set_ylim([ylim_min, ylim_max])
        else:
            ax.set_ylim(ylim)

    def _customize_epoch_plot(
        self,
        ax,
        title,
        x_scale="linear",
        y_scale="log",
        fontweight="normal",
        facecolor=None,
    ):
        """[summary]
        Args:
            ax ([type]): [description]
            ax2 ([type]): [description]
            title ([type]): [description]
            fontweight (str, optional): [description]. Defaults to "normal".
            facecolor ([type], optional): [description]. Defaults to None.
        """
        ax.set_title(title, fontweight=fontweight)

        # Set Background Color
        if facecolor is not None:
            ax.set_facecolor(self.color_summary_plots)

        # Start x axis from zero and make it tight
        ax.set_xlim(left=0)

        ax.set_xlim(right=self.epoch_tracking["iteration"].iloc[-1])

        # Set Scales
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        # Fix Legend
        lines, labels = ax.get_legend_handles_labels()

        plot_labels = []
        for line, label in zip(lines, labels):
            plot_labels.append("{0}: ({1:.2E})".format(label, line.get_ydata()[-1]))
        ax.get_legend().remove()
        ax.legend(lines, plot_labels)
