"""Class for tracking the Gradient Norm."""

from backboard.quantities.quantity import ByproductQuantity
from backboard.quantities.utils_quantities import _root_sum_of_squares


class GradNorm(ByproductQuantity):
    """Gradient Norm Quantitiy Class."""

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
        if self.is_active(global_step):
            grad_norm = [p.grad.data.norm(2).item() for p in params]
            self.output[global_step]["grad_norm"] = grad_norm

            if self._verbose:
                print(
                    f"[Step {global_step}] GradNorm:"
                    + " {_root_sum_of_squares(grad_norm):.4f}"
                )
