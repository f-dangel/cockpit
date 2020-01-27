"""Test for the quadratic fit and the computation of the local step size alpha."""

import numpy as np

from backboard.utils.cockpit_utils import _fit_quadratic, _get_alpha


def test_determinstic_fit_min():
    """Verify that fitting works in the noise free case."""
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
    """Verify that fitting works in the noise free case."""
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
    """Verify that fitting works in the noise free case."""
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
    """Verify that fitting works in the noise free case."""
    # Observations
    t = 1  # where
    fs = [1, 0.5]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [0, -1]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, 0, -1 / 2])

    alpha = _get_alpha(mu, t)

    assert alpha >= 1e12


def test_stochastic_fit_min():
    """Verify that fitting works in the noisy case.

    Here there are two contradictory observations, but one with a very high
    uncertainty, that it should not trust."""
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
    """Verify that fitting works in the noisy case.

    Here there are two contradictory observations, but one with a very high
    uncertainty, that it should not trust."""
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
    """Verify that fitting works in the noisy case.

    Here there are two contradictory observations, but one with a very high
    uncertainty, that it should not trust."""
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
