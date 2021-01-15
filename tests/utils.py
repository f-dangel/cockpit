"""Utility functions for running tests."""

import sys

import numpy

from cockpit.deepobs import set_deepobs_seed


def hotfix_deepobs_argparse():
    """Truncate command line arguments from pytest call to make DeepOBS arparse work.

    TODO Think about good alternatives.
    """
    sys.argv = sys.argv[:1]


def report_nonclose_values(x, y, atol=1e-8, rtol=1e-5):
    """Report non-close values.

    Note:
        ``numpy.allclose`` and ``torch.allclose`` don't always
        seem to match when using the same parameters for ``atol`` and
        ``rtol``. Maybe related to data types, but I could not find
        a helpful reference.
        Therefore it may happen that nonclose values are reported,
        while the tests pass at the same time.
    """
    x_numpy = x.data.cpu().numpy().flatten()
    y_numpy = y.data.cpu().numpy().flatten()

    close = numpy.isclose(x_numpy, y_numpy, atol=atol, rtol=rtol)
    where_not_close = numpy.argwhere(numpy.logical_not(close))
    for idx in where_not_close:
        x, y = x_numpy[idx], y_numpy[idx]
        print(f"{x} versus {y}. Ratio of {y/x}")


def has_negative(tensor, verbose=True):
    """Does a tensor contain negative entries."""
    tensor_numpy = tensor.data.cpu().numpy().flatten()
    where_negative = numpy.argwhere(tensor_numpy < 0)

    if verbose:
        for idx in where_negative:
            value = float(tensor_numpy[idx])
            print(f"Encountered negative value: {value:.5f}")

    negative_count = len(where_negative)
    negative = negative_count != 0

    if verbose and negative:
        print(f"Encountered {negative_count} negative values")

    return negative


def has_nans(tensor, verbose=True):
    """Does a tensor contain NaNs."""
    tensor_numpy = tensor.data.cpu().numpy().flatten()
    where_nan = numpy.argwhere(tensor_numpy != tensor_numpy)

    nan_count = len(where_nan)
    nan = nan_count != 0

    if verbose and nan:
        print(f"Encountered {nan_count} NaNs")

    return nan


def has_zeros(tensor, verbose=True):
    """Does a tensor contain zeros."""
    tensor_numpy = tensor.data.cpu().numpy().flatten()
    where_zero = numpy.argwhere(tensor_numpy == 0.0)

    zero_count = len(where_zero)
    zero = zero_count != 0

    if verbose and zero:
        print(f"Encountered {zero_count} zeros")

    return zero


def set_up_problem(tproblem_cls, batch_size=5, seed=None, l2_reg=0.0):
    """Create DeepOBS problem with neural network, and set to train mode."""
    if seed is not None:
        set_deepobs_seed(seed)

    tproblem = tproblem_cls(batch_size, l2_reg=l2_reg)

    tproblem.set_up()
    tproblem.train_init_op()

    return tproblem
