"""Class for tracking the EB criterion for early stopping."""

import warnings

from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_SumGradSquared


class EarlyStopping(SingleStepQuantity):
    """Evidence-based (EB) early-stopping criterion.

    Note: Proposed in

        - Mahsereci, M., Balles, L., Lassner, C., & Hennig, P.,
          Early stopping without a validation set (2017).
    """

    def __init__(self, track_schedule, verbose=False, epsilon=1e-5):
        """Initialize.

        Args:
            epsilon (float): Stabilization constant. Defaults to 0.0.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._epsilon = epsilon
        warnings.warn("StoppingCriterion only applies to SGD without momentum.")

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
        """Compute the EB early stopping criterion.

        Evaluates the left hand side of Equ. 7 in

        - Mahsereci, M., Balles, L., Lassner, C., & Hennig, P.,
          Early stopping without a validation set (2017).

        If this value exceeds 0, training should be stopped.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        B = get_batch_size(global_step)

        grad_squared = self._fetch_grad(params, aggregate=True) ** 2

        # compensate BackPACK's 1/B scaling
        sgs_compensated = (
            B ** 2
            * self._fetch_sum_grad_squared_via_batch_grad_transforms(
                params, aggregate=True
            )
        )

        diag_variance = (sgs_compensated - B * grad_squared) / (B - 1)

        snr = grad_squared / (diag_variance + self._epsilon)

        return 1 - B * snr.mean()
