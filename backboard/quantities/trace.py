"""Class for tracking the Trace of the Hessian."""

from backboard.quantities.quantity import Quantity
from backpack import extensions


class Trace(Quantity):
    """Trace Quantitiy Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return (
            [extensions.DiagHessian()]
            if global_step % self._track_interval == 0
            else []
        )

    def compute(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params (method): Function to access the parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if global_step % self._track_interval == 0:
            self.output[global_step]["trace"] = [
                p.diag_h.sum().item() for p in params() if p.requires_grad
            ]
        else:
            pass
