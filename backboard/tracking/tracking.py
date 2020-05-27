"""This is an extension of the CockpitTracker class.

It contains all the iter_tracking functions, defining the computation needed for the
logged quantity.
These functions are then used in either track_before, track_after or track_epoch
of the CockpitTracker class.

self is a CockpitTracker."""

import time
from math import sqrt

import numpy as np
import torch
from scipy.sparse.linalg import eigsh

from .utils_ev import HVPLinearOperator
from .utils_tracking import (_acute_angle_test_sin, _combine_batch_l2,
                             _combine_grad, _combine_grad_batch,
                             _exact_variance, _fit_quadratic, _get_alpha,
                             _get_batch_size, _inner_product_test_width,
                             _layerwise_dot_product, _norm_test_radius)


def track_f(self, batch_loss, point):
    """Tracks the function value at the (start or end) point.

    Args:
        batch_loss (float): Average loss over a batch
        point (str): Either "0" or "1" to signify that we are iter_tracking the
            starting or end point of an iteration.
    """
    self.iter_tracking["f" + point].append(batch_loss.item())


def track_var_f(self, batch_losses, point):
    """Tracks the variance of the function value at the (start or end) point.

    Args:
        batch_losses (list): List of individual losses in a batch.
        point (str): Either "0" or "1" to signify that we are iter_tracking the
            starting or end point of an iteration.
    """
    self.iter_tracking["var_f" + point].append(batch_losses.var().item())


def track_df(self, point):
    """Tracks the projected gradient (onto the search direction) at the
    (start or end) point.

    Args:
        point (str): Either "0" or "1" to signify that we are iter_tracking the
            starting or end point of an iteration.
    """
    self.iter_tracking["df" + point].append(
        _layerwise_dot_product(
            self.search_dir,
            [p.grad.data for p in self.parameters() if p.requires_grad],
        )
    )


def track_var_df(self, point):
    """Tracks the variance of the projected gradient (onto the search direction)
    at the (start or end) point.

    Args:
        point (str): Either "0" or "1" to signify that we are iter_tracking the
            starting or end point of an iteration.
    """
    self.iter_tracking["var_df" + point].append(
        _exact_variance(
            [p.grad_batch.data for p in self.parameters() if p.requires_grad],
            self.search_dir,
        )
    )


def track_grad_norms(self):
    """Tracks the L2 norm of the current gradient."""
    self.iter_tracking["grad_norms"].append(
        [p.grad.data.norm(2).item() for p in self.parameters() if p.requires_grad]
    )


def track_dtravel(self, learning_rate):
    """Tracks the distance traveled in each iteration.

    It is very important that this function is computed AFTER iter_tracking
    grad_norms.
    TODO This definition only applies to SGD without Momentum.

    Args:
        learning_rate (float): Learning rate used in this step.
    """
    self.iter_tracking["dtravel"].append(
        [el * learning_rate for el in self.iter_tracking["grad_norms"][-1]]
    )


def track_trace(self):
    """Tracks the trace of the Hessian."""
    self.iter_tracking["trace"].append(
        [p.diag_h.sum().item() for p in self.parameters() if p.requires_grad]
    )


def track_ev(self, batch_loss):
    """Track the max (and possibly min) eigenvalue of the Hessian.

    Args:
        batch_loss (float): Average loss over a batch
    """
    trainable_params = [p for p in self.parameters() if p.requires_grad]
    HVP = HVPLinearOperator(
        batch_loss, trainable_params, grad_params=[p.grad for p in trainable_params],
    )
    eigvals = eigsh(HVP, k=1, which="LA", return_eigenvectors=False)

    self.iter_tracking["max_ev"].append(np.float64(eigvals))


def track_d2init(self):
    """Tracks the L2 distance of the current parameters to their init."""
    self.iter_tracking["d2init"].append(
        [
            (init - p).norm(2).item()
            for init, p in zip(self.p_init, self.parameters())
            if p.requires_grad
        ]
    )


def track_alpha(self):
    """Tracks the effective relative step size.

    It is measured as were we "step" on the local 1D quadratic approximation.
    An alpha of 0 means that the step was to the minimum of the parabola.
    An alpha of -1 means we stayed at the same position of the quadratic.
    An alpha of 1 means we stepped on the other side of the quadratic.

    If we cannot make a quadratic fit (most likely due to variances of 0) alpha
    is set to None (via the `_get_alpha` function).
    """
    # We need to find the size of the step taken,
    # since dtravel can be a list, we need to aggregate it
    if type(self.iter_tracking["dtravel"][-1]) is list:
        t = sqrt(sum(t * t for t in self.iter_tracking["dtravel"][-1]))
    else:
        t = self.iter_tracking["dtravel"][-1]

    # Fit a noise-informed quadratic approximation to the obersvation of
    # function value and projected gradient plus their variance.
    mu = _fit_quadratic(
        t,
        [self.iter_tracking["f0"][-1], self.iter_tracking["f1"][-1]],
        [sum(self.iter_tracking["df0"][-1]), sum(self.iter_tracking["df1"][-1]),],
        [self.iter_tracking["var_f0"][-1], self.iter_tracking["var_f1"][-1]],
        [
            sum(self.iter_tracking["var_df0"][-1]),
            sum(self.iter_tracking["var_df1"][-1]),
        ],
    )

    # Get the relative (or local) step size
    self.iter_tracking["alpha"].append(_get_alpha(mu, t))


def track_global_norm_test_radius(self):
    """Track norm test radius for the concatenated network parameters."""
    B = _get_batch_size(self.parameters())
    batch_l2 = _combine_batch_l2(self.parameters())
    grad = _combine_grad(self.parameters())

    radius = _norm_test_radius(B, batch_l2, grad)

    print("Global norm test: ", radius)
    time.sleep(1)

    self.iter_tracking["global_norm_test_radius"].append(radius)


def track_norm_test_radius(self):
    """Track the ball radius around the expected risk gradient.

    .. note::
        The norm test radius `r` is not additive over layers.
    """

    def parameter_norm_test_radius(p):
        B = _get_batch_size(self.parameters())

        radius = _norm_test_radius(B, p.batch_l2, p.grad)

        print("Param norm test: ", radius)
        time.sleep(1)

        return radius

    self.iter_tracking["norm_test_radius"].append(
        [parameter_norm_test_radius(p) for p in self.parameters() if p.requires_grad]
    )


def track_global_inner_product_test_width(self):
    """Track inner product test width for the concatenated network parameters."""
    B = _get_batch_size(self.parameters())
    grad_batch = _combine_grad_batch(self.parameters())
    grad = _combine_grad(self.parameters())

    width = _inner_product_test_width(B, grad_batch, grad)
    print("Global inner product test: ", width)
    time.sleep(1)

    self.iter_tracking["global_inner_product_test_width"].append(width)


def track_inner_product_test_width(self):
    """Track the band width orthogonal to the expected risk gradient.

    .. note::
        The inner product test width `w` is not additive over layers.
    """

    def parameter_inner_product_test_width(p):
        B = _get_batch_size(self.parameters())

        width = _inner_product_test_width(B, p.grad_batch, p.grad)

        print("Param inner product test: ", width)
        time.sleep(1)

        return width

    self.iter_tracking["inner_product_test_width"].append(
        [
            parameter_inner_product_test_width(p)
            for p in self.parameters()
            if p.requires_grad
        ]
    )


def track_global_acute_angle_test_sin(self):
    """Track acute angle test sinus for the concatenated network parameters."""
    B = _get_batch_size(self.parameters())
    batch_l2 = _combine_batch_l2(self.parameters())
    grad_batch = _combine_grad_batch(self.parameters())
    grad = _combine_grad(self.parameters())

    sin = _acute_angle_test_sin(B, grad_batch, batch_l2, grad)

    print("Global acute angle test: ", sin)
    time.sleep(1)

    self.iter_tracking["global_acute_angle_test_sin"].append(sin)


def track_acute_angle_test_sin(self):
    """Track the angle sinus between mini-batch and expected risk gradient.

    .. note::
        The acute angle test sinus `s` is not additive over layers.
    """

    def parameter_acute_angle_test_sin(p):
        B = _get_batch_size(self.parameters())

        sin = _acute_angle_test_sin(B, p.grad_batch, p.batch_l2, p.grad)

        print("Param acute angle test: ", sin)
        time.sleep(1)

        return sin

    self.iter_tracking["acute_angle_test_sin"].append(
        [
            parameter_acute_angle_test_sin(p)
            for p in self.parameters()
            if p.requires_grad
        ]
    )
