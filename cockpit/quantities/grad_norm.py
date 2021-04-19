"""Class for tracking the Gradient Norm."""

from cockpit.quantities.quantity import ByproductQuantity


class GradNorm(ByproductQuantity):
    """Quantitiy Class for tracking the norm of the mean gradient."""

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the gradient norm at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            torch.Tensor: The quantity's value.
        """
        return [p.grad.data.norm(2).item() for p in params]
