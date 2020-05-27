"""Utility functions for the CockpitTracker."""

import itertools
import os
import warnings
from math import sqrt

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
        fs_var (float): Variances of function values at 0 and t.
        dfs_var (float): Variances of projected gradients at 0 and t.

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

    Args:
        mu (list): Parameters of the quadratic fit.
        t (float): Step size (taken).

    Returns:
        float: Local effective step size.
    """
    # If we couldn't compute a quadratic approx., return None.
    if mu is None:
        return None
    else:
        # get alpha_bar (the step size that is "the other side")
        alpha_bar = -mu[1] / mu[2]

        # scale everything on a standard quadratic (i.e. step of 0 = minimum)
        return (2 * t) / alpha_bar - 1


def _normalize(v):
    return v / v.norm()


def _combine_grad(parameters):
    """Construct grad for concatenation of flattened parameters."""
    return torch.cat([p.grad.flatten() for p in parameters if p.requires_grad])


def _combine_grad_batch(parameters):
    """Construct grad_batch for concatenation of flattened parameters."""
    flat_grad_batch = [
        p.grad_batch.reshape(p.shape[0], -1) for p in parameters if p.requires_grad
    ]
    return torch.cat(flat_grad_batch, dim=1)


def _combine_batch_l2(parameters):
    """Construct batch_l2 for concatenation of flattened parameters."""
    return sum(p.batch_l2 for p in parameters if p.requires_grad)


def _norm_test_radius(B, batch_l2, grad):
    """Ball radius around the expected risk gradient.

    Introduced in:
        Sample size selection in optimization methods for machine learning
        by Richard H. Byrd, Gillian M. Chin, Jorge Nocedal & Yuchen Wu
        (Mathematical Programming volume 134, pages 127–155 (2012))

    Let `g_B, g_P` denote the gradient of the mini-batch and expected risk,
    respectively. The norm test proposes to satisfy
        `E( || g_B - g_P ||² / || g_P ||² ) ≤ r²`
    for some user-specified radius `r`.

    A practical form with mini-batch gradients `gₙ` is given by
        `(1 / (|B| (|B| - 1))) * ∑ₙ || gₙ - g_B ||² / || g_B ||² ≤ r²`.

    We track the square root of the above equation's LHS.
    """

    square_radius = 1 / (B - 1) * (B * (batch_l2.sum() / grad.norm(2) ** 2) - 1)
    return sqrt(square_radius)


def _inner_product_test_width(B, grad_batch, grad):
    """Band width orthogonal to the expected risk gradient.

    Introduced in:
        Adaptive Sampling Strategies for Stochastic Optimization
        by Raghu Bollapragada, Richard Byrd, Jorge Nocedal
        (2017)

    Let `g_B, g_P` denote the gradient of the mini-batch and expected risk,
    respectively. The inner product test proposes to satisfy
        `E( gᵀ_B g_P / || g_P ||² ) ≤ w²`
    for some user-specified band width `w`.

    A practical form with mini-batch gradients `gₙ` is given by
        `(1 / (|B| (|B| - 1))) * ∑ₙ ( gᵀₙ g_B / || g_B ||² - 1 )² ≤ w²`.

    We track the square root of the above equation's LHS.
    """

    projection = (
        torch.einsum("bi,i->b", grad_batch.reshape(B, -1), grad.reshape(-1))
        / grad.norm(2) ** 2
    )
    square_width = 1 / (B - 1) * (B * (projection ** 2).sum() - 1)
    return sqrt(square_width)


def _acute_angle_test_sin(B, grad_batch, batch_l2, grad):
    """Angle sinus between mini-batch and expected risk gradient.

    Although defined differently in terms of inaccessible quantities,
    the estimation from a mini-batch is equivalent to the orthogonality
    test in Bollapragada (2017).

    Introduced in:
        A Dynamic Sampling Adaptive-SGD Method for Machine Learning
        by Achraf Bahamou, Donald Goldfarb
        (2020)

    Let `g_B, g_P` denote the gradient of the mini-batch and expected risk,
    respectively. The acute angle test proposes to satisfy
        `E( sin (∠( g_B, g_P )² ) ≤ s²`
    for some user-specified sinus `s`.

    A practical form with mini-batch gradients `gₙ` is given by
        `(1 / (|B| (|B| - 1))) * (
                                  ∑ₙ || gₙ ||² / || g_B ||²
                                  - 2 B ∑ₙ ( gᵀₙ g_B )² / || g_B ||⁴
                                  +B
                                  )                                 ≤ s²`.

    We track the square root of the above equation's LHS.
    """
    summand1 = B * batch_l2.sum() / grad.norm(2) ** 2

    flat_grad_batch = grad_batch.reshape(B, -1)
    flat_grad = grad.reshape(-1)

    summand2 = (
        -2
        * B
        * (torch.einsum("bi,i->b", flat_grad_batch, flat_grad) ** 2).sum()
        / grad.norm(2) ** 4
    )

    square_sin = 1 / (B - 1) * (summand1 + summand2 + 1)
    return sqrt(square_sin)


def _get_batch_size(parameters):
    """Infer batch size from saved BackPACK extensions.

    `B` is reconstructed from the gradient L2 norm.
    """
    for p in parameters:
        if p.requires_grad:
            B = p.batch_l2.shape[0]
            return B

    raise ValueError("No parameter with requires_grad=True found")
