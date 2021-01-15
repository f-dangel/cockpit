"""Utility functions for cockpit benchmark."""

from functools import lru_cache

from deepobs.pytorch.testproblems import (
    cifar10_3c3d,
    cifar100_3c3d,
    cifar100_allcnnc,
    fmnist_2c2d,
    fmnist_mlp,
    mnist_2c2d,
    mnist_logreg,
    mnist_mlp,
    quadratic_deep,
)


@lru_cache()
def get_train_size(testproblem):
    """Return number of samples in training set."""
    tproblem_cls_from_str = {
        "cifar10_3c3d": cifar10_3c3d,
        "cifar100_3c3d": cifar100_3c3d,
        "cifar100_allcnnc": cifar100_allcnnc,
        "fmnist_2c2d": fmnist_2c2d,
        "fmnist_mlp": fmnist_mlp,
        "mnist_2c2d": mnist_2c2d,
        "mnist_logreg": mnist_logreg,
        "mnist_mlp": mnist_mlp,
        "quadratic_deep": quadratic_deep,
    }
    tproblem_cls = tproblem_cls_from_str[testproblem]

    return _get_train_size(tproblem_cls)


def _get_train_size(tproblem_cls):
    """Return number of samples in training set."""
    batch_size = 1

    tproblem = tproblem_cls(batch_size=batch_size)
    tproblem.set_up()

    return _get_train_steps_per_epoch(tproblem) * batch_size


def _get_train_steps_per_epoch(tproblem):
    """Return number of mini-batches in the train set."""
    tproblem.train_init_op()

    steps = 0

    try:
        while True:
            tproblem._get_next_batch()
            steps += 1
    except StopIteration:
        return steps
    except Exception as e:
        raise RuntimeError(f"Failed to detect steps per epoch: {e}")
