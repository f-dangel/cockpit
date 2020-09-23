"""Utility functions for cockpit benchmark."""

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
    tproblem = tproblem_cls(batch_size=1)
    tproblem.set_up()
    tproblem.train_init_op()

    train_size = 0

    try:
        while True:
            (x, _) = tproblem._get_next_batch()
            train_size += x.shape[0]
    except StopIteration:
        pass

    return train_size
