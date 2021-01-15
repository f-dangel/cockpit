"""Plotting functionality for the paper experiments."""

import os
import subprocess

import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save


class TikzExport:
    """Handle matplotlib export to TikZ."""

    def __init__(self, extra_axis_parameters=None):
        """
        Note:
        -----
        Extra axis parameters are inserted in alphabetical order.
        By prepending 'z' to the style, it will be inserted last.
        Like that, you can overwrite the previous axis parameters
        in 'zmystyle'.
        """
        if extra_axis_parameters is None:
            extra_axis_parameters = {"zmystyle"}

        self.extra_axis_parameters = extra_axis_parameters

    def save_fig(
        self,
        out_file,
        fig=None,
        png_preview=True,
        tex_preview=True,
        override_externals=True,
        post_process=True,
    ):
        """Save matplotlib figure as TikZ. Optional PNG out.

        Create the directory if it does not exist.
        """
        if fig is not None:
            self.set_current(fig)

        tex_file = self._add_extension(out_file, "tex")

        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)

        if png_preview:
            png_file = self._add_extension("{}-preview".format(out_file), "png")
            plt.savefig(png_file, bbox_inches="tight")

        tikz_save(
            tex_file,
            override_externals=override_externals,
            extra_axis_parameters=self.extra_axis_parameters,
        )

        if post_process is True:
            self.post_process(out_file)

        if tex_preview is True:
            tex_preview_file = self._add_extension("{}-preview".format(out_file), "tex")
            with open(tex_file, "r") as f:
                content = "".join(f.readlines())

            preamble = r"""\documentclass[tikz]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{amssymb}
\usetikzlibrary{shapes,
                pgfplots.groupplots,
                shadings,
                calc,
                arrows,
                backgrounds,
                colorbrewer,
                shadows.blur}

% customize "zmystyle" as you wish
 \pgfkeys{/pgfplots/zmystyle/.style={
         % legend pos = north east,
         % xmin=1, xmax=20,
         % ymin = 1, ymax = 1.2,
         % title = {The title},
     }
 }

\begin{document}
"""

            postamble = r"\end{document}"
            preview_content = preamble + content + postamble

            with open(tex_preview_file, "w") as f:
                f.write(preview_content)

            subprocess.run(["pdflatex", "-output-directory", out_dir, tex_preview_file])

    def save_subplots(
        self,
        out_path,
        names,
        fig=None,
        png_preview=True,
        tex_preview=True,
        override_externals=True,
        post_process=True,
    ):
        """Save subplots of figure into single TikZ figures."""
        if fig is None:
            fig = plt.gcf()

        axes = self.axes_as_individual_figs(fig)

        for name, subplot in zip(names, axes):
            assert len(subplot.get_axes()) == 1

            out_file = os.path.join(out_path, name)
            self.save_fig(
                out_file,
                fig=subplot,
                png_preview=png_preview,
                tex_preview=tex_preview,
                override_externals=override_externals,
                post_process=post_process,
            )

    @staticmethod
    def set_current(fig):
        plt.figure(fig.number)

    def post_process(self, tikz_file):
        """Remove from matplotlib2tikz export what should be configurable."""
        file = self._add_extension(tikz_file, "tex")
        with open(file, "r") as f:
            content = f.readlines()

        content = self._remove_linewidths(content)
        content = self._remove_some_arguments(content)

        joined_content = "".join(content)

        with open(file, "w") as f:
            f.write(joined_content)

    @staticmethod
    def _add_extension(filename, extension, add_to_filename=None):
        if add_to_filename is None:
            add_to_filename = ""
        return "{}{}.{}".format(filename, add_to_filename, extension)

    @staticmethod
    def _remove_linewidths(lines):
        """Remove line width specifications."""
        linewidths = [
            r"ultra thick",
            r"very thick",
            r"semithick",
            r"thick",
            r"very thin",
            r"ultra thin",
            r"thin",
        ]
        new_lines = []
        for line in lines:
            for width in linewidths:
                line = line.replace(width, "")
            new_lines.append(line)
        return new_lines

    @staticmethod
    def _remove_some_arguments(lines):
        """Remove lines containing certain specifications."""
        # remove lines containing these specifications
        to_remove = [
            r"legend cell align",
            # r"legend style",
            r"x grid style",
            r"y grid style",
            # r"tick align",
            # r"tick pos",
            r"ytick",
            r"xtick",
            r"yticklabels",
            r"xticklabels",
            # "ymode",
            r"log basis y",
        ]

        for pattern in to_remove:
            lines = [line for line in lines if pattern not in line]

        return lines

    @staticmethod
    def axes_as_individual_figs(fig):
        """Return a list of figures, each containing a single axes.

        `fig` is messed up during this procedure as the axes are being removed
        and inserted into other figures.

        Note: MIGHT BE UNSTABLE
        -----
        https://stackoverflow.com/questions/6309472/
        matplotlib-can-i-create-axessubplot-objects-then-add-them-to-a-figure-instance

        Axes deliberately aren't supposed to be shared between different figures now.
        As a workaround, you could do this fig2._axstack.add(fig2._make_key(ax), ax),
        but it's hackish and likely to change in the future.
        It seems to work properly, but it may break some things.
        """
        fig_axes = fig.get_axes()

        # breaks fig
        for ax in fig_axes:
            fig.delaxes(ax)

        fig_list = []
        for ax in fig_axes:
            new_fig = plt.figure()
            new_fig._axstack.add(new_fig._make_key(ax), ax)
            new_fig.axes[0].change_geometry(1, 1, 1)
            fig_list.append(new_fig)

        return fig_list
