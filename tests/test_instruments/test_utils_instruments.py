"""Tests for the utility functions of the instruments."""

import matplotlib.pyplot as plt
import pytest

from cockpit.instruments.utils_instruments import _compute_plot_limits


# Test Plot Limits
def _simple_problem():
    x = [0, 1]
    y = [-2, 2]

    x_interval = x[-1] - x[0]
    y_interval = y[-1] - y[0]

    default_mpl_margin = 0.05

    return x, y, x_interval, y_interval, default_mpl_margin


x, y, x_interval, y_interval, default_mpl_margin = _simple_problem()

xlim_tests = [
    (
        None,
        [
            x[0] - x_interval * default_mpl_margin,
            x[1] + x_interval * default_mpl_margin,
        ],
    ),
    (
        [None, None],
        [
            x[0] - x_interval * default_mpl_margin,
            x[1] + x_interval * default_mpl_margin,
        ],
    ),
    ("tight", [x[0], x[1]]),
    (["tight", "tight"], [x[0], x[1]]),
]
ylim_tests = [
    (
        None,
        [
            y[0] - y_interval * default_mpl_margin,
            y[1] + y_interval * default_mpl_margin,
        ],
    ),
    (
        [None, None],
        [
            y[0] - y_interval * default_mpl_margin,
            y[1] + y_interval * default_mpl_margin,
        ],
    ),
    ("tight", [y[0], y[1]]),
    (["tight", "tight"], [y[0], y[1]]),
]


@pytest.mark.parametrize("xlim,expected_xlim", xlim_tests)
@pytest.mark.parametrize("ylim,expected_ylim", ylim_tests)
def test_compute_plot_limits(xlim, ylim, expected_xlim, expected_ylim):
    """Test whether the plot limits are computed correctly.

    Args:
        xlim (None, str, list): Defining the limits of the x-axis.
        ylim (None, str, list): Defining the limits of the y-axis.
        expected_xlim (list): A list of floats of the expected x-limits.
        expected_ylim (list): A list of floats of the expected y-limits.
    """
    # Create a simple plot with known limits
    x, y, _, _, _ = _simple_problem()

    plt.clf()
    plt.plot(x, y)
    ax = plt.gca()

    o_xlim, o_ylim = _compute_plot_limits(ax, xlim, ylim)

    assert o_xlim == expected_xlim
    assert o_ylim == expected_ylim

    plt.clf()
