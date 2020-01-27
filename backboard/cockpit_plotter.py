"""Plotting Part of the Cockpit"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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

    def plot(self, draw=True, save=False, save_append=None):
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

        self.grid_spec = self.fig.add_gridspec(6, self.n_layers + 1)

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
        for i in range(self.n_layers):
            self._plot_trace(self.grid_spec[2, i], i)
        self._plot_trace(self.grid_spec[2, -1])

        # # grad norm gauge
        # for i in range(self.n_layers):
        #     self._plot_grad_norm(self.grid_spec[2, i], i)
        # self._plot_grad_norm(self.grid_spec[2, -1])

        # d2init gauge
        for i in range(self.n_layers):
            self._plot_dist(self.grid_spec[3, i], i)
        self._plot_dist(self.grid_spec[3, -1])

        # Performance plot
        han, leg = self._plot_perf(self.grid_spec[4, :])

        # Epoch Plot
        self._plot_epoch(self.grid_spec[5, :])

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

        if draw:
            plt.show()
            print("Cockpit-Plot shown...")
        else:
            plt.close(self.fig)
            print("Cockpit-Plot drawn...")
        plt.pause(0.01)
        if save:
            self.fig.savefig(self.log_path + save_append + ".png")
            print("Cockpit-Plot saved...")

    def save_plot(self):
        """Saves the last cockpit plot to image file."""
        self.fig.savefig(self.log_path + ".png")
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
        self.iter_tracking = self.iter_tracking.reset_index().rename(
            columns={"index": "iteration"}
        )

        self.epoch_tracking = pd.DataFrame.from_dict(data["epoch_tracking"])
        self.epoch_tracking = self.epoch_tracking.reset_index().rename(
            columns={"index": "epoch"}
        )

        self._process_tracking_results()

    def _process_tracking_results(self):
        # expand lists
        self.n_layers = 1
        for (columnName, columnData) in self.iter_tracking.items():
            if isinstance(columnData[0], list):
                temp = self.iter_tracking[columnName].apply(pd.Series)
                self.n_layers = max(self.n_layers, temp.shape[1])
                temp = temp.rename(columns=lambda x: columnName + "_layer_" + str(x))
                # Aggregate for all layers
                if columnName in [
                    "df0",
                    "df1",
                    "var_df_0",
                    "var_df_1",
                    "df_var_0",
                    "df_var_1",
                    "trace",
                ]:
                    # Aggregate via sum
                    temp.loc[:, columnName] = temp.sum(axis=1)
                elif columnName in ["grad_norms", "d2init", "dtravel"]:
                    # Aggregate via root of sum of squares:
                    temp.loc[:, columnName] = temp.pow(2).sum(axis=1).pow(1 / 2)
                else:
                    print(
                        "**Don't know how I should aggregate "
                        + str(columnName)
                        + " for all layers"
                    )
                self.iter_tracking = self.iter_tracking.drop([columnName], axis=1)
                self.iter_tracking = pd.concat([self.iter_tracking[:], temp[:]], axis=1)
        # TODO: We currently divide the trace by 7850 since this is the number
        # of parameters for mnist logreg. We need to change this for different
        # testproblems
        self.iter_tracking["avg_cond"] = (
            self.iter_tracking["max_ev"] / self.iter_tracking["trace"] * 7850
        )

    def _plot_epoch(self, gridspec):
        """Creates a plot of variables that are tracked every epoch

        Args:
            gridspec ([gridspec]): Gridspec to plot into
        """
        # Plot Settings
        x_quan = "epoch"
        y_quan = "lr"
        y_quan2 = "train_acc"
        y_quan3 = "valid_acc"
        x_scale = "linear"
        y_scale = "log"
        y_scale2 = "linear"
        title = "Epoch Plot"

        # Plotting
        ax = self.fig.add_subplot(gridspec)

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
        self._customize_epoch_plot(
            ax,
            ax2,
            title,
            x_scale=x_scale,
            y_scale=y_scale,
            y_scale2=y_scale2,
            epochs=self.epoch_tracking.shape[0] - 1,
            fontweight="bold",
            facecolor=self.color_summary_plots,
        )

    def _plot_perf(self, gridspec):
        """Create the loss vs. iteration plot."""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "f0"
        x_scale = "linear"
        y_scale = "log"
        title = "Performance Plot"

        # Plotting
        ax = self.fig.add_subplot(gridspec)
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
            extend_factors=[0.0, 0.05],
        )

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
        title = "f gauge for all layers"
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

    def _plot_alpha(self, gridspec):
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
        ax = self.fig.add_subplot(gridspec)
        # Plot unit parabola
        x = np.linspace(xlim[0], xlim[1], 100)
        y = x ** 2
        ax.plot(x, y, linewidth=2)

        # Alpha Histogram
        ax2 = ax.twinx()
        # All alphas
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
        # Just from last plot
        sns.distplot(
            self.iter_tracking["alpha"][-self.iter_per_plot :],
            ax=ax2,
            # norm_hist=True,
            fit=stats.norm,
            kde=False,
            color=sns.color_palette()[1],
            fit_kws={"color": sns.color_palette()[1]},
            hist_kws={"linewidth": 0, "alpha": 0.65},
            label="since last plot",
        )

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
        (mu_all, _) = stats.norm.fit(self.iter_tracking["alpha"])
        (mu_last, _) = stats.norm.fit(
            self.iter_tracking["alpha"][-self.iter_per_plot :]
        )
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(
            [
                "{0} ($\mu=${1:.2f})".format(labels2[0], mu_all),  # noqa: W605
                "{0} ($\mu=${1:.2f})".format(labels2[1], mu_last),  # noqa: W605
            ]
        )

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
            title = "df gauge for all layers"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "df gauge for layer " + str(layer)
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

    def _plot_trace(self, gridspec, layer="all"):
        """Plot the trace"""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "trace"
        x_scale = "linear"
        y_scale = "log"
        n = "" if isinstance(layer, str) else "_layer_" + str(layer)

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan + n] = (
            self.iter_tracking[y_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
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

        # Customize Plot
        if isinstance(layer, str):
            title = "Trace for all Layers"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "Trace for Layer " + str(layer)
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

    def _plot_grad_norm(self, gridspec, layer="all"):
        """Plot the gradient norm gauge"""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "grad_norms"
        x_scale = "linear"
        y_scale = "linear"
        n = "" if isinstance(layer, str) else "_layer_" + str(layer)

        # Compute derived quantities
        self.iter_tracking["EMA_" + y_quan + n] = (
            self.iter_tracking[y_quan + n].ewm(alpha=self.EMA_span, adjust=False).mean()
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

        # Customize Plot
        if isinstance(layer, str):
            title = "Grad Norm Gauge for all Layers"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "Grad Norm Gauge for Layer " + str(layer)
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
        x_quan = "min_ev"
        y_quan = "max_ev"
        x_scale = "log"
        y_scale = "log"
        title = "Min-Max Eigenvalue"
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

    def _plot_dist(self, gridspec, layer="all"):
        """Plot the distance."""
        # Plot Settings
        x_quan = "iteration"
        y_quan = "d2init"
        y_quan2 = "dtravel"
        x_scale = "linear"
        y_scale = "linear"
        n = "" if isinstance(layer, str) else "_layer_" + str(layer)

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
            title = "Distance Gauge for all Layers"
            fontweight = "bold"
            facecolor = self.color_summary_plots
        else:
            title = "Distance Gauge for Layer " + str(layer)
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
            ax.sey_xlim(xlim)

        if ylim is None:
            ax.set_ylim([ylim_min, ylim_max])
        else:
            ax.set_ylim(ylim)

    def _customize_epoch_plot(
        self,
        ax,
        ax2,
        title,
        x_scale="linear",
        y_scale="log",
        y_scale2="linear",
        epochs=None,
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

        # Set Accuracies limites to 0 and 1
        ax2.set_ylim([0, 1])

        # Set Background Color
        if facecolor is not None:
            ax.set_facecolor(self.color_summary_plots)

        # Start x axis from zero and make it tight
        ax.set_xlim(left=0)
        ax2.set_xlim(left=0)

        if epochs is not None:
            ax.set_xlim(right=epochs)
            ax2.set_xlim(right=epochs)

        # Set Scales
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax2.set_yscale(y_scale2)

        # Fix Legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        plot_labels = []
        for line, label in zip(lines + lines2, labels + labels2):
            if label == "lr":
                plot_labels.append("{0}: ({1:.2E})".format(label, line.get_ydata()[-1]))
            else:
                plot_labels.append("{0}: ({1:.2%})".format(label, line.get_ydata()[-1]))
        ax.get_legend().remove()
        ax2.get_legend().remove()
        ax.legend(lines + lines2, plot_labels)
