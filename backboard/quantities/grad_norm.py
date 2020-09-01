"""Class for tracking the Gradient Norm."""

from backboard.quantities.quantity import Quantity


class GradNorm(Quantity):
    """Gradient Norm Quantitiy Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def compute(self, global_step, params, batch_loss):
        """Evaluate the gradient norm at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            torch.Tensor: The quantity's value.
        """
        if global_step % self._track_interval == 0:
            self.output[global_step]["grad_norm"] = [
                p.grad.data.norm(2).item() for p in params() if p.requires_grad
            ]
        else:
            pass
