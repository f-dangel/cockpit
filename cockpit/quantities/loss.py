"""Class for tracking the loss."""

from cockpit.quantities.quantity import ByproductQuantity


class Loss(ByproductQuantity):
    """Loss Quantity class tracking the mini-batch training loss during training."""

    def _compute(self, global_step, params, batch_loss):
        """Track the loss at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Mini-batch loss at the current iteration.
        """
        return batch_loss.item()
