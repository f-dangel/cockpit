"""Variants of the ``cifar10_3c3d`` DeepOBS problem."""

import torch

from backboard.experiments.utils import replace
from deepobs.pytorch.datasets.cifar10 import cifar10
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
