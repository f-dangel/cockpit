"""Class for tracking the loss."""

from cockpit.quantities.quantity import ByproductQuantity


class Loss(ByproductQuantity):
    """Loss Quantity Class."""

    # TODO Rewrite to use parent class track method
    def track(self, global_step, params, batch_loss):
        """Track the loss at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            loss = batch_loss.item()
            self.output[global_step]["mini_batch_loss"] = loss

            if self._verbose:
                print(f"[Step {global_step}] Loss: {loss:.4f}")
