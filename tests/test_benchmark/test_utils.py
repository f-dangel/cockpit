"""Tests for ``backboard.benchmark.utils``."""

import pytest

from backboard.benchmark.utils import get_train_size
from backboard.utils import fix_deepobs_data_dir

train_sizes = {
    "quadratic_deep": 1000,
    "mnist_2c2d": 50000,
    "mnist_logreg": 50000,
    "mnist_mlp": 50000,
    "fmnist_2c2d": 50000,
    "fmnist_mlp": 50000,
    "cifar10_3c3d": 40000,
    "cifar100_allcnnc": 40000,
    "cifar100_3c3d": 40000,
}


@pytest.mark.parametrize("tproblem_cls", list(train_sizes.keys()))
def test_get_train_size(tproblem_cls):
    """Test function to determine number of samples in train set."""
    fix_deepobs_data_dir()

    assert get_train_size(tproblem_cls) == train_sizes[tproblem_cls]
