"""Utility functions for the CockpitTracker."""

import copy
import itertools
import os
import warnings

import numpy as np
import torch


def _check_optimizer(optimizer):
    """Check whether we currently support this optimizer.

    Note, we currently only support vanilla SGD since the search direction and
    update magnitude is hard-coded.

    Args:
        optimizer (torch.optim): A PyTorch optimizer
    """
    if optimizer.__class__.__name__ == "SGD":
        if optimizer.param_groups[0]["momentum"] != 0:
            warnings.warn(
                "Warning: You are using SGD with momentum. Computation of "
                "parameter update magnitude and search direction is "
                "probably incorrect!",
                stacklevel=2,
            )
    else:
        warnings.warn(
            "Warning: You are using an optimizer, with an unknown parameter "
            "update. Computation of parameter update magnitude and search "
            "direction is probably incorrect!",
            stacklevel=2,
        )


def _prepare_logpath(logpath):
    """Prepare the logpath by creating it if necessary

    Args:
        logpath (str): The path where the logs should be stored
    """
    logdir, logfile = os.path.split(logpath)
    os.makedirs(logdir, exist_ok=True)


def _init_tracking(per_iter_quants, per_epoch_quants):
    """Initialize dicts for our tracking quantities.

    This is a helper function that initializes each entry of the dict as an
    empty list. This makes it easier to append to when tracking.

    Args:
        per_iter_quants (list): List of quantities that are tracked per iteration
        per_epoch_quants (list): List for quantities that are tracked per epoch

    Returns:
        [tuple]: A tuple of initialized dictionaries.
    """
    # Create empty dicts for iter and epoch quantites
    iter_tracking = dict()
    epoch_tracking = dict()
    # fill them with empty lists
    for quant in per_iter_quants:
        iter_tracking[quant] = []
    for quant in per_epoch_quants:
        epoch_tracking[quant] = []

    return iter_tracking, epoch_tracking


def _layerwise_dot_product(x_s, y_s):
    """Computes the dot product of two parameter vectors layerwise.
    Args:
        x_s (list): First list of parameter vectors.
        y_s (list): Second list of parameter vectors.
    Returns:
        prod: 1-D list of scalars. Each scalar is a dot product of one layer.
    """
    return [torch.sum(x * y).item() for x, y in zip(x_s, y_s)]


def _exact_variance(grads, search_dir):
    """Given a batch of individual gradients, it computes the exact variance
    of their projection onto the search direction.

    Args:
        grads (list): List of individual gradients
        search_dir (list): List of search direction
    """
    # swap order of first two axis, so now it is:
    # grads[batch_size, layer, parameters]
    grads = [
        [i for i in element if i is not None]
        for element in list(itertools.zip_longest(*grads))
    ]
    proj_grad = []
    for grad in grads:
        proj_grad.append(_layerwise_dot_product(search_dir, grad))
    return np.var(np.array(proj_grad), axis=0, ddof=1).tolist()


def _fit_quadratic(t, fs, dfs, fs_var, dfs_var):
    """Fit a quadratic, given (noisy) observations of two function values and
    two projected gradients

    Args:
        t (float): Position of second observation.
        fs (float): Function values at 0 and t.
        dfs (float): Projected gradients at 0 and t.
        fs_var (list): Variances of function values at 0 and t.
        dfs_var (list): Variances of projected gradients at 0 and t.

    Returns:
        np.array: Parameters of the quadratic fit.
    """
    # The first observation is always at 0, the second at t
    # Observation matrix (f(0), then f(t), then f'(0) and f'(t))
    # The matrix shows how we observe the parameters of the quadratic
    # i.e. for f(0) we see (0)**2 *w1 + (0)*w2 + w3
    # and for f'(t) we see 2*(t)*w1 + w2

    Phi = np.array([[1, 0, 0], [1, t, t ** 2], [0, 1, 0], [0, 1, 2 * t]]).T

    try:
        # Covariance matrix
        lambda_inv = np.linalg.inv(np.diag(fs_var + dfs_var))

        # Maximum Likelihood Estimation
        mu = np.linalg.inv(Phi @ lambda_inv @ Phi.T) @ Phi @ lambda_inv @ (fs + dfs)
    except np.linalg.LinAlgError:
        if 0 in dfs_var:
            warnings.warn("The variance of df is 0, couldn't compute alpha.")
        elif 0 in fs_var:
            warnings.warn("The variance of f is 0, couldn't compute alpha.")
        else:
            warnings.warn("Couldn't compute alpha for some unknown reason.")
        mu = None

    return mu


def _get_alpha(mu, t):
    """Compute the local step size alpha.
    It will be expressed in terms of the local quadratic fit. A local step size
    of 1, is equal to `stepping on the other side of the quadratic`, while a 
    step size of 0 means `stepping to the minimum`.

    If 

    Args:
        mu (list): Parameters of the quadratic fit.
        t (float): Step size (taken).

    Returns:
        float: Local effective step size.
    """
    # If we couldn't compute a quadratic approx., return None.
    if mu is None:
        return None
    elif mu[2] < 0:
        # Concave setting: Since this means that it is still going downhill
        # where we stepped to, we will log it as -1.
        return -1
    else:
        # get alpha_bar (the step size that is "the other side")
        alpha_bar = -mu[1] / mu[2]

        # scale everything on a standard quadratic (i.e. step of 0 = minimum)
        return (2 * t) / alpha_bar - 1


def _normalize(v):
    """Normalize a vector.

    Args:
        v (torch.Tensor): Some vector

    Returns:
        torch.Tensor: Normalized vector of norm 1.
    """
    return v / v.norm()


def _trim_dict(dicty):
    """Returns a trimmed dictionary such that it each value containst a list of the same
    size.

    We assume that all list have either the size n or n+1. The last elements of
    all lists of size n+1 should be trimmed.

    Args:
        dicty (dict): A dictionary where each value is a list.

    Returns:
        dict: A dictionary with equally sized lists as values
    """
    # Create a deepcopy, so that poping an item does not change the original
    trimmed_dicty = copy.deepcopy(dicty)

    # Create dict where the values are the lenghts of the lists
    len_dicty = {key: len(value) for key, value in trimmed_dicty.items()}

    # get a sorted list of the unique lenghts
    lengths = list(set(len_dicty.values()))
    lengths.sort()

    # We don't have to do anything if the sizes are correct. E.g. if we break
    # from the loop while not tracking the current iteration
    if len(lengths) == 1:
        return trimmed_dicty

    # Make sure that it has at most 2 different lengths
    assert (
        len(lengths) == 2
    ), "Couldn't trim the list as it has more than two different lengths."
    # make sure that the difference between the two largest ones is exactly 1
    assert (
        lengths[-1] - lengths[-2] == 1
    ), "Couldn't trim the list as the difference between the sizes was more than one."

    for key, value in len_dicty.items():
        # pop last element of those lists that are too large
        if value == lengths[-1]:
            trimmed_dicty[key].pop()

    return trimmed_dicty
