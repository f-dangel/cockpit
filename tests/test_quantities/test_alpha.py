"""Tests for the quadratic fit and the computation of the local step size alpha."""

import numpy as np
import pytest

from backboard.quantities.alpha import (
    AlphaExpensive,
    AlphaOptimized,
    _fit_quadratic,
    _get_alpha,
)
from tests.test_quantities.utils import compare_quantities, get_output_sgd_test_runner


def test_determinstic_fit_min():
    r"""Verify that fitting works in the noise free case.

    This is an example where we "walked to the minimum" \_
    """
    # Observations
    t = 1  # where
    fs = [1, 0.5]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [-1, 0]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, -1, 1 / 2])

    alpha = _get_alpha(mu, t)

    assert np.isclose(alpha, 0)


def test_determinstic_fit_other_side():
    r"""Verify that fitting works in the noise free case.

    This is an example where we "walked to the other side" \ /
    """
    # Observations
    t = 1  # where
    fs = [1, 1]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [-1, 1]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, -1, 1])

    alpha = _get_alpha(mu, t)

    assert np.isclose(alpha, 1)


def test_determinstic_fit_linear():
    r"""Verify that fitting works in the noise free case.

    This is an example where it is still going downhill \
                                                         \
    """
    # Observations
    t = 1  # where
    fs = [1, 0]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [-1, -1]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, -1, 0])

    alpha = _get_alpha(mu, t)

    assert np.isclose(alpha, -1)


def test_determinstic_fit_concave():
    r"""Verify that fitting works in the noise free case.

    This is an example where we start at a local maxima _
                                                         \.
    """
    # Observations
    t = 1  # where
    fs = [1, 0.5]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [0, -1]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, 0, -1 / 2])

    alpha = _get_alpha(mu, t)

    assert alpha == -1


def test_stochastic_fit_min():
    r"""Verify that fitting works in the noisy case.

    Here there are two contradictory observations, but one with a very high
    uncertainty, that it should not trust.

    This is an example where we "walked to the minimum" \_
    """
    # Observations
    t = 1  # where
    fs = [1, 0.5]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [-1, 1]  # derivative
    dfs_var = [1e-10, 1e5]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, -1, 1 / 2])

    alpha = _get_alpha(mu, t)

    assert np.isclose(alpha, 0)


def test_stochastic_fit_other_side():
    r"""Verify that fitting works in the noisy case.

    Here there are two contradictory observations, but one with a very high
    uncertainty, that it should not trust.

    This is an example where we "walked to the other side" \ /
    """
    # Observations
    t = 1  # where
    fs = [1, 0]  # loss value
    fs_var = [1e-10, 1e5]  # variance of loss
    dfs = [-1, 1]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, -1, 1])

    alpha = _get_alpha(mu, t)

    assert np.isclose(alpha, 1)


def test_small_step():
    r"""Verify that fitting works in the noisy case.

    Here there are two contradictory observations, but one with a very high
    uncertainty, that it should not trust.

    This is an example where we took a very small step, geting the same loss
    and derivative
    """
    # Observations
    t = 1e-10  # where
    fs = [1, 1]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [-1, -1]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, -1, 0])

    alpha = _get_alpha(mu, t)

    assert np.isclose(alpha, -1)


TESTPROBLEMS = [
    "mnist_logreg",
    # "fmnist_2c2d",
    "mnist_mlp",
    "fmnist_logreg",
    "fmnist_mlp",
    # "mnist_2c2d",
    # "cifar10_3c3d",
]

TRACK_INTERVAL = 2


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration_expensive(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Integration test for expensive alpha quantity.

    Computes the effective local step size alpha during a short training.
    Note: This test only verifies that the computation passes.
    """
    quantities = [AlphaExpensive(TRACK_INTERVAL, verbose=True)]

    return get_output_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
    )[0]


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration_optimized(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Integration test for expensive alpha quantity.

    Computes the effective local step size alpha during a short training.
    Note: This test only verifies that the computation passes.
    """
    quantities = [AlphaOptimized(TRACK_INTERVAL, verbose=True)]

    return get_output_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
    )[0]


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_expensive_matches_optimized_separate_runs(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Compare results of expensive and optimized alpha."""
    quantity1 = AlphaOptimized(TRACK_INTERVAL, verbose=True)
    quantity2 = AlphaExpensive(TRACK_INTERVAL, verbose=True)

    compare_quantities(
        [quantity1, quantity2],
        testproblem,
        separate_runs=True,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
    )


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_expensive_matches_optimized_joint_runs(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Compare results of expensive and optimized alpha."""
    quantities = [
        AlphaOptimized(TRACK_INTERVAL, verbose=True),
        AlphaExpensive(TRACK_INTERVAL, verbose=True),
    ]

    compare_quantities(
        quantities,
        testproblem,
        separate_runs=False,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
    )
