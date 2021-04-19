"""Class for tracking the Mean Gradient Signal to Noise Ration (GSNR)."""


from backpack.extensions import BatchGrad

from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook_SumGradSquared


class MeanGSNR(SingleStepQuantity):
    """Quantitiy Class for the mean gradient signal-to-noise ratio (GSNR).

    Note: Mean gradient signal-to-noise ratio as defined by

        - Liu, J., et al.
          Understanding Why Neural Networks Generalize Well Through GSNR of
          Parameters (2020).
          https://arxiv.org/abs/2001.07384
    """

    def __init__(self, track_schedule, verbose=False, epsilon=1e-5):
        """Initialize.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
            epsilon (float): Stabilization constant. Defaults to 1e-5.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._epsilon = epsilon

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.should_compute(global_step):
            ext.append(BatchGrad())

        return ext

    def extension_hooks(self, global_step):
        """Return list of BackPACK extension hooks required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            [callable]: List of required BackPACK extension hooks for the current
                iteration.
        """
        hooks = []

        if self.should_compute(global_step):
            hooks.append(BatchGradTransformsHook_SumGradSquared())

        return hooks

    def _compute(self, global_step, params, batch_loss):
        """Track the mean GSNR.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Mean GSNR of the current iteration.
        """
        return self._compute_gsnr(global_step, params, batch_loss).mean().item()

    def _compute_gsnr(self, global_step, params, batch_loss):
        """Compute gradient signal-to-noise ratio.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Mean GSNR of the current iteration.
        """
        grad_squared = self._fetch_grad(params, aggregate=True) ** 2
        sum_grad_squared = self._fetch_sum_grad_squared_via_batch_grad_transforms(
            params, aggregate=True
        )

        batch_size = get_batch_size(global_step)

        return grad_squared / (
            batch_size * sum_grad_squared - grad_squared + self._epsilon
        )
