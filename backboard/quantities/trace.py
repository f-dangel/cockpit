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
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if global_step % self._track_interval == 0:
            trace = [p.diag_h.sum().item() for p in params]
            self.output[global_step]["trace"] = trace

            if self._verbose:
                print(f"Trace: {sum(trace):.4f}")

        else:
            pass
