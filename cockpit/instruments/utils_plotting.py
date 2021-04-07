"""Utility functions for the CockpitPlotter."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def _extract_problem_info(source):
    """Split the logpath to identify test problem, data set, etc.

    Args:
        source (Cockpit or str): ``Cockpit`` instance, or string containing the
                path to a .json log produced with ``Cockpit.write``, where
                information will be fetched from.

    Returns:
        [dict]: Dictioniary of logpath, testproblem, optimizer, etc.
    """
    if isinstance(source, str):
        # Split logpath if possible
        try:
            dicty = {
                "logpath": source + ".json",
                "optimizer": source.split("/")[-3],
                "testproblem": source.split("/")[-4],
                "dataset": source.split("/")[-4].split("_", 1)[0],
                "model": source.split("/")[-4].split("_", 1)[1],
            }
        except Exception:
            dicty = {
                "logpath": source + ".json",
                "optimizer": "",
                "testproblem": "",
                "dataset": "",
                "model": "",
            }
    else:
        # Source is Cockpit instance
        dicty = {
            "logpath": "",
            "optimizer": source._optimizer_name,
            "testproblem": "",
            "dataset": "",
            "model": "",
        }
    return dicty


def legend():
    """Creates the legend of the whole cockpit, combining the individual instruments."""
    pass


def _alpha_cmap(color, ncolors=256):
    """Create a Color map that goes from transparant to a given color.

    Args:
        color (tuple): A matplotlib-compatible color.
        ncolors (int, optional): Number of "steps" in the colormap.
            Defaults to 256.

    Returns:
        [matplotlib.cmap]: A matplotlib colormap
    """
    color_array = np.array(ncolors * [list(color)])

    # change alpha values
    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
    # create a colormap object
    cmap = LinearSegmentedColormap.from_list(name="alpha_cmap", colors=color_array)

    return cmap
