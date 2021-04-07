"""Utility functions to investigate optimizers.

Some quantities either require computation of the optimizer update step, or are only
defined for certain optimizers.
"""

import numpy as np
import torch


class ComputeStep:
    """Update step computation from BackPACK quantities for different optimizers.

    Note:
        The ``.grad`` attribute cannot be used to compute update steps as this code
        is invoked as hook during backpropagation at a time where the ``.grad`` field
        has not yet been updated with the latest gradients.
    """

    @staticmethod
    def compute_update_step(optimizer, parameter_ids):
        """Compute an optimizer's update step.

         Args:
             optimizer (torch.optim.Optimizer): A PyTorch optimizer.
             parameter_ids ([id]): List of parameter ids for which the updates are
                 computed.

        Returns:
             dict: Mapping between parameters and their updates. Keys are parameter ids
                 and items are ``torch.Tensor``s representing the update.

        Raises:
             NotImplementedError: If the optimizer's update step is not implemented.
        """
        if ComputeStep.is_sgd_default_kwargs(optimizer):
            return ComputeStep.update_sgd_default_kwargs(optimizer, parameter_ids)

        raise NotImplementedError

    @staticmethod
    def is_sgd_default_kwargs(optimizer):
        """Return whether the input is momentum-free SGD with default values.

        Args:
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.

        Returns:
            bool: Whether the input is momentum-free SGD with default values.
        """
        if not isinstance(optimizer, torch.optim.SGD):
            return False

        for group in optimizer.param_groups:
            if not np.isclose(group["weight_decay"], 0.0):
                return False

            if not np.isclose(group["momentum"], 0.0):
                return False

            if not np.isclose(group["dampening"], 0.0):
                return False

            if not group["nesterov"] is False:
                return False

        return True

    @staticmethod
    def update_sgd_default_kwargs(optimizer, parameter_ids):
        """Return the update of momentum-free SGD with default values.

        Args:
            optimizer (torch.optim.SGD): Zero-momentum default SGD.
            parameter_ids ([id]): List of parameter ids for which the updates are
                 computed.

        Returns:
            dict: Mapping between parameters and their updates. Keys are parameter ids
                and items are ``torch.Tensor``s representing the update.
        """
        updates = {}

        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) in parameter_ids:
                    lr = group["lr"]
                    updates[id(p)] = -lr * p.grad_batch.sum(0).detach()

                    if len(updates.keys()) == len(parameter_ids):
                        return updates

        assert len(updates.keys()) == len(
            parameter_ids
        ), "Could not compute step for all specified parameters"

        return updates
