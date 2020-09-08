"""Utility Functions for the Quantities and the Tracking in General."""

import numpy as np
import torch


def _update_dicts(master_dict, update_dict):
    """Merge dicts of dicts by updating dict a with dict b.

    Args:
        master_dict (dict): [description]
        update_dict (dict): [description]
    """
    for key, value in update_dict.items():
        for subkey, subvalue in value.items():
            master_dict[key][subkey] = subvalue


def _unreduced_loss_hotfix(batch_loss):
    """A simple hotfix for the unreduced loss values.

    For the quadratic_deep problem of DeepOBS, the unreduced losses are a matrix
    and should be averaged over the second axis.

    Args:
        batch_loss (torch.Tensor): Mini-batch loss from current step, with the
            unreduced losses as an attribute.

    Returns:
        torch.Tensor: (Averaged) unreduced losses.
    """
    batch_losses = batch_loss._unreduced_loss
    if len(batch_losses.shape) == 2:
        batch_losses = batch_losses.mean(1)
    return batch_losses


def _layerwise_dot_product(x_s, y_s):
    """Computes the dot product of two parameter vectors layerwise.

    Args:
        x_s (list): First list of parameter vectors.
        y_s (list): Second list of parameter vectors.

    Returns:
        torch.Tensor: 1-D list of scalars. Each scalar is a dot product of one layer.
    """
    return [torch.sum(x * y).item() for x, y in zip(x_s, y_s)]


def _root_sum_of_squares(list):
    """Returns the root of the sum of squares of a given list.

    Args:
        list (list): A list of floats

    Returns:
        [float]: Root sum of squares
    """
    return sum((el ** 2 for el in list)) ** (0.5)


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

    close = np.isclose(x_numpy, y_numpy, atol=atol, rtol=atol)
    where_not_close = np.argwhere(np.logical_not(close))
    for idx in where_not_close:
        x, y = x_numpy[idx], y_numpy[idx]
        print(f"{x} versus {y}. Ratio of {y/x}")


def has_negative(tensor, verbose=True):
    """Does a tensor contain negative entries."""
    tensor_numpy = tensor.data.cpu().numpy().flatten()
    where_negative = np.argwhere(tensor_numpy < 0)

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
    where_nan = np.argwhere(tensor_numpy != tensor_numpy)

    nan_count = len(where_nan)
    nan = nan_count != 0

    if verbose and nan:
        print(f"Encountered {nan_count} NaNs")

    return nan


def has_zeros(tensor, verbose=True):
    """Does a tensor contain zeros."""
    tensor_numpy = tensor.data.cpu().numpy().flatten()
    where_zero = np.argwhere(tensor_numpy == 0.0)

    zero_count = len(where_zero)
    zero = zero_count != 0

    if verbose and zero:
        print(f"Encountered {zero_count} zeros")

    return zero
