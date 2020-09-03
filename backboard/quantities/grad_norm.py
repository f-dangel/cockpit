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
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            torch.Tensor: The quantity's value.
        """
        if global_step % self._track_interval == 0:
            grad_norm = [p.grad.data.norm(2).item() for p in params]
            self.output[global_step]["grad_norm"] = grad_norm

            if self._verbose:
                print(f"Grad norm: {sum(grad_norm):.4f}")
        else:
            pass
