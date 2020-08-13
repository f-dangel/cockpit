"""Tests for the quadratic fit and the computation of the local step size alpha."""

import numpy as np

from backboard.tracking.utils_tracking import _fit_quadratic, _get_alpha


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
                                                         \
    """
    # Observations
    t = 1  # where
    fs = [1, 0.5]  # loss value
    fs_var = [1e-10, 1e-10]  # variance of loss
    dfs = [0, -1]  # derivative
    dfs_var = [1e-10, 1e-10]  # variance of derivative

    mu = _fit_quadratic(t, fs, dfs, fs_var, dfs_var)

    assert np.allclose(mu, [1, 0, -1 / 2])

    print(mu)

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
