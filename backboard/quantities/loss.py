"""Class for tracking the loss."""

from backboard.quantities.quantity import ByproductQuantity


class Loss(ByproductQuantity):
    """Loss Quantity Class."""

    def compute(self, global_step, params, batch_loss):
        """Track the loss at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            self.output[global_step]["mini_batch_loss"] = batch_loss.item()

            if self._verbose:
                print(f"Loss: {batch_loss.item():.4f}")
