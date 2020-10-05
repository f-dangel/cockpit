"""Tests for the quadratic fit and the computation of the local step size alpha."""

import numpy as np
import pytest

from backboard.quantities.alpha import (
    AlphaExpensive,
    AlphaOptimized,
    _fit_quadratic,
    _get_alpha,
)

from tests.test_quantities.test_runner import run_sgd_test_runner
from tests.utils import hotfix_deepobs_argparse, set_deepobs_seed


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
    "fmnist_2c2d",
    "mnist_mlp",
    "fmnist_logreg",
    "fmnist_mlp",
    "mnist_2c2d",
    # "cifar10_3c3d",
]

TRACK_INTERVAL = 2


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration_alpha_expensive(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0
):
    """Integration test for expensive alpha quantity.

    Computes the effective local step size alpha during a short training.
    Note: This test only verifies that the computation passes.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [AlphaExpensive(TRACK_INTERVAL, verbose=True)]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )

    alpha = quantities[0].output
    return alpha


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration_alpha_optimized(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0
):
    """Integration test for expensive alpha quantity.

    Computes the effective local step size alpha during a short training.
    Note: This test only verifies that the computation passes.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [AlphaOptimized(TRACK_INTERVAL, verbose=True)]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )

    alpha = quantities[0].output
    return alpha


def compare_outputs(output1, output2, rtol=5e-4, atol=1e-5):
    """Compare outputs of two quantities."""
    assert len(list(output1.keys())) == len(
        list(output2.keys())
    ), "Different number of entries"

    for key in output1.keys():
        if isinstance(output1[key], dict):
            compare_outputs(output1[key], output2[key])
        else:
            val1, val2 = output1[key], output2[key]
            if isinstance(val1, float) and isinstance(val2, float):
                assert np.isclose(val1, val2, atol=atol, rtol=rtol)
            else:
                raise NotImplementedError("No comparison available for this data type.")


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_expensive_matches_optimized_alpha_separate_runs(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0
):
    """Compare results of expensive and optimized alpha."""
    alpha_optimized = test_integration_alpha_optimized(
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )

    alpha_expensive = test_integration_alpha_expensive(
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )

    compare_outputs(alpha_optimized, alpha_expensive)


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_expensive_matches_optimized_alpha_joint_run(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0
):
    """Integration test for expensive alpha quantity.

    Computes the effective local step size alpha during a short training.
    Note: This test only verifies that the computation passes.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [
        AlphaOptimized(TRACK_INTERVAL, verbose=True),
        AlphaExpensive(TRACK_INTERVAL, verbose=True),
    ]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )

    alpha_optimized = quantities[0].output
    alpha_expensive = quantities[1].output

    compare_outputs(alpha_optimized, alpha_expensive)
