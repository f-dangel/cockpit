"""Utility functions for running tests."""

import random
import sys

import numpy
import torch


def set_deepobs_seed(seed=0):
    """Set all seeds used by DeepOBS."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def hotfix_deepobs_argparse():
    """Truncate command line arguments from pytest call to make DeepOBS arparse work.

    TODO Think about good alternatives.
    """
    sys.argv = sys.argv[:1]


def report_nonclose_values(x, y, atol=1e-8, rtol=1e-5):
    """Report non-close values.

    Note: ``numpy.allclose`` and ``torch.allclose`` don't always
    seem to match when using the same parameters for ``atol`` and
    ``rtol``. Maybe related to data types, but I could not find
    a helpful reference.
    Therefore it may happen that nonclose values are reported,
    while the tests pass at the same time.
    """
    x_numpy = x.data.cpu().numpy().flatten()
    y_numpy = y.data.cpu().numpy().flatten()

    close = numpy.isclose(x_numpy, y_numpy, atol=atol, rtol=atol)
    where_not_close = numpy.argwhere(numpy.logical_not(close))
    for idx in where_not_close:
        x, y = x_numpy[idx], y_numpy[idx]
        print(f"{x} versus {y}. Ratio of {y/x}")


def has_negative(tensor, verbose=True):
    """Does a tensor contain negative entries."""
    negative = False
    negative_count = 0

    def is_negative(value):
        return value < 0

    for value in tensor.flatten():
        if is_negative(value):
            negative = True
            negative_count += 1

            if verbose:
                print(f"Encountered negative value: {value:.5f}")

    if verbose and negative:
        print(f"Encountered {negative_count} negative values")

    return negative


def has_nans(tensor, verbose=True):
    """Does a tensor contain NaNs."""
    nans = False
    nan_count = 0

    def is_nan(value):
        return value != value

    for value in tensor.flatten():
        if is_nan(value):
            nans = True
            nan_count += 1

    if verbose and nans:
        print(f"Encountered {nan_count} NaNs")

    return nans


def has_zeros(tensor, verbose=True):
    """Does a tensor contain zeros."""
    zeros = False
    zeros_count = 0

    def is_zero(value):
        return value == 0.0

    for value in tensor.flatten():
        if is_zero(value):
            zeros = True
            zeros_count += 1

    if verbose and zeros:
        print(f"Encountered {zeros_count} zeros")

    return zeros
