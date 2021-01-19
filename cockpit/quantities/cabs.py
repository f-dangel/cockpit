"""Class for tracking the CABS criterion for adaptive batch size."""

from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_SumGradSquared


class CABS(SingleStepQuantity):
    """CABS criterion, coupling adaptive batch sizes and learning rates.

    Only applies to SGD without momentum.

    Note: Proposed in

        - Balles, L., Romero, J., & Hennig, P.,
          Coupling adaptive batch sizes with learning rates (2017).
    """

    def __init__(self, track_schedule, verbose=False, lr=1.0):
        """Initialize.

        Args:
            lr (float): Learning rate. Defaults to 1.0.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
        """
        super().__init__(track_schedule, verbose=verbose)

        self.set_lr(lr)

    def set_lr(self, lr):
        """Set value for current learning rate."""
        self._lr = lr

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.should_compute(global_step):
            ext.append(BatchGradTransforms_SumGradSquared())

        return ext

    def _compute(self, global_step, params, batch_loss):
        """Compute the CABS rule. Return suggested batch size.

        Evaluates Equ. 22 of

        - Balles, L., Romero, J., & Hennig, P.,
          Coupling adaptive batch sizes with learning rates (2017).

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        B = get_batch_size(global_step)
        lr = self._lr

        grad_squared = self._fetch_grad(params, aggregate=True) ** 2
        # # compensate BackPACK's 1/B scaling
        sgs_compensated = (
            B ** 2
            * self._fetch_sum_grad_squared_via_batch_grad_transforms(
                params, aggregate=True
            )
        )

        return lr * (sgs_compensated - B * grad_squared).sum() / (B * batch_loss.item())
