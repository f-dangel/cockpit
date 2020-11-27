"""Variants of the ``cifar10_3c3d`` DeepOBS problem."""

import torch
from torchvision import datasets

from cockpit.deepobs.experiments.utils import register, replace
from deepobs import config
from deepobs.pytorch.datasets.cifar10 import cifar10
from deepobs.pytorch.datasets.dataset import DataSet
from deepobs.pytorch.testproblems import cifar10_3c3d
from deepobs.pytorch.testproblems.testproblems_modules import net_cifar10_3c3d


class _cifar10_3c3dact(cifar10_3c3d):
    """``cifar10_3c3d`` replacing ReLU activations in 3c3d net."""

    def set_up(self):
        """Set up the vanilla CNN test problem on Cifar-10."""
        self.data = cifar10(self._batch_size)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = net_cifar10_3c3d(num_outputs=10)
        self._replace_relu(self.net)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    @staticmethod
    def _replace_relu(module):
        """Replace all occurrences of `ReLU`."""
        raise NotImplementedError()


class cifar10_3c3dsig(_cifar10_3c3dact):
    """``cifar10_3c3d`` replacing ReLU activations by Sigmoids in 3c3d net."""

    @staticmethod
    def _replace_relu(module):
        """Replace all occurrences of `ReLU` by `Sigmoids`."""

        def trigger(mod):
            return isinstance(mod, torch.nn.ReLU)

        def make_new(mod):
            return torch.nn.Sigmoid()

        replace(module, trigger, make_new)


class cifar10_3c3dtanh(_cifar10_3c3dact):
    """``cifar10_3c3d`` replacing ReLU activations by Tanh in 3c3d net."""

    @staticmethod
    def _replace_relu(module):
        """Replace all occurrences of `ReLU` by `Tanh`."""

        def trigger(mod):
            return isinstance(mod, torch.nn.ReLU)

        def make_new(mod):
            return torch.nn.Tanh()

        replace(module, trigger, make_new)


def make_cifar10transform(transform, transform_name):
    """Create cifar10 dataloader with specific data-preprocessing."""

    class cifar10transform(DataSet):
        """Dataset that uses the specified pre-processing transformation."""

        def __init__(self, batch_size, train_eval_size=10000):
            self._train_eval_size = train_eval_size
            self._transform_name = transform_name
            self._transform = transform
            self._name = f"cifar10{self._transform_name}"
            super().__init__(batch_size)

        def _make_train_and_valid_dataloader(self):
            train_dataset = datasets.CIFAR10(
                root=config.get_data_dir(),
                train=True,
                download=True,
                transform=self._transform,
            )
            valid_dataset = datasets.CIFAR10(
                root=config.get_data_dir(),
                train=True,
                download=True,
                transform=self._transform,
            )

            train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
                train_dataset, valid_dataset
            )

            return train_loader, valid_loader

        def _make_test_dataloader(self):
            test_dataset = datasets.CIFAR10(
                root=config.get_data_dir(),
                train=False,
                download=True,
                transform=self._transform,
            )

            return self._make_dataloader(test_dataset, sampler=None)

    # modify name
    cifar10transform.__name__ = cifar10transform.__name__.replace(
        "transform", transform_name
    )

    return cifar10transform


def make_cifar10transform_3c3d(transform, transform_name):
    """Create and register cifar10 3c3d testproblem with specific data pre-processing"""

    dataset_cls = make_cifar10transform(transform, transform_name)

    class cifar10transform_3c3d(cifar10_3c3d):
        """3c3d testproblem on transformed cifar10 dataset."""

        _transform_name = transform_name
        _transform = transform

        def set_up(self):
            super().set_up()

            # overwrite data
            self.data = dataset_cls(self._batch_size)

    cifar10transform_3c3d.__name__ = cifar10transform_3c3d.__name__.replace(
        "transform", transform_name
    )

    register(cifar10transform_3c3d)

    return cifar10transform_3c3d


def make_cifar10transform_3c3dsig(transform, transform_name):
    """Create and register cifar10 3c3dsig testproblem with specific pre-processing"""

    dataset_cls = make_cifar10transform(transform, transform_name)

    class cifar10transform_3c3dsig(cifar10_3c3dsig):
        """3c3d testproblem on transformed cifar10 dataset."""

        _transform_name = transform_name
        _transform = transform

        def set_up(self):
            super().set_up()

            # overwrite data
            self.data = dataset_cls(self._batch_size)

    cifar10transform_3c3dsig.__name__ = cifar10transform_3c3dsig.__name__.replace(
        "transform", transform_name
    )

    register(cifar10transform_3c3dsig)

    return cifar10transform_3c3dsig


def make_cifar10transform_3c3dtanh(transform, transform_name):
    """Create and register cifar10 3c3dtanh testproblem with specific pre-processing"""

    dataset_cls = make_cifar10transform(transform, transform_name)

    class cifar10transform_3c3dtanh(cifar10_3c3dtanh):
        """3c3d testproblem on transformed cifar10 dataset."""

        _transform_name = transform_name
        _transform = transform

        def set_up(self):
            super().set_up()

            # overwrite data
            self.data = dataset_cls(self._batch_size)

    cifar10transform_3c3dtanh.__name__ = cifar10transform_3c3dtanh.__name__.replace(
        "transform", transform_name
    )

    register(cifar10transform_3c3dtanh)

    return cifar10transform_3c3dtanh
