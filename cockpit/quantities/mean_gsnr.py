"""Class for tracking the Mean Gradient Signal to Noise Ration (GSNR)."""

import torch

from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_quantities import (
    has_nans,
    has_negative,
    has_zeros,
    report_nonclose_values,
)
from cockpit.quantities.utils_transforms import BatchGradTransforms_SumGradSquared

# TODO Move to tests
ATOL = 1e-5
RTOL = 5e-4


class MeanGSNR(SingleStepQuantity):
    """Mean GSNR Quantitiy Class.

    Mean gradient signal-to-noise ratio.

    Reference:
        Understanding Why Neural Networks Generalize Well Through
        GSNR of Parameters
        by Jinlong Liu, Guoqing Jiang, Yunzhi Bai, Ting Chen, Huayan Wang
        (2020)
    """

    def __init__(self, track_schedule, verbose=False, epsilon=1e-5):
        """Initialize.

        Args:
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
            ext.append(BatchGradTransforms_SumGradSquared())

        return ext

    def _compute(self, global_step, params, batch_loss):
        """Track the mean GSNR.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        return self._compute_gsnr(global_step, params, batch_loss).mean()

    def _compute_gsnr(self, global_step, params, batch_loss):
        """Compute gradient signal-to-noise ratio.

        Args:
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        grad_squared = self._fetch_grad(params, aggregate=True) ** 2
        sum_grad_squared = self._fetch_sum_grad_squared_via_batch_grad_transforms(
            params, aggregate=True
        )

        batch_size = get_batch_size(global_step)

        return grad_squared / (
            batch_size * sum_grad_squared - grad_squared + self._epsilon
        )

    # TODO Move to tests
    def __run_check(self, global_step, params, batch_loss):
        """Check if variance is non-negative and hence GSNR is not NaN."""

        def _compute_gsnr_from_batch_grad(params):
            """Gradient signal-to-noise ratio.

            Implement equation (25) in liu2020understanding, recursively defined via
            the prose between Equation (1) and (2).
            """
            batch_grad = self._fetch_batch_grad(params, aggregate=True)
            batch_size = get_batch_size(global_step)

            rescaled_batch_grad = batch_size * batch_grad

            grad_first_moment_squared = (rescaled_batch_grad).mean(0) ** 2
            grad_second_moment = (rescaled_batch_grad ** 2).mean(0)
            grad_variance = grad_second_moment - grad_first_moment_squared

            if has_negative(grad_variance + self._epsilon):
                raise ValueError("Gradient variances from batch_grad are negative.")

            if has_zeros(grad_variance + self._epsilon):
                raise ValueError("Gradient variances + ε has zeros.")

            return grad_first_moment_squared / (grad_variance + self._epsilon)

        # sanity check 1: Both GSNRs do not contain NaNs
        gsnr_from_batch_grad = _compute_gsnr_from_batch_grad(params)
        assert not has_nans(gsnr_from_batch_grad), "GSNR from batch_grad has NaNs"

        gsnr_from_sum_grad_squared = self._compute_gsnr(global_step, params, batch_loss)
        assert not has_nans(
            gsnr_from_sum_grad_squared
        ), "GSNR from sum_grad_squared has NaNs"

        # sanity check 2: Both GSNRs match
        report_nonclose_values(
            gsnr_from_sum_grad_squared, gsnr_from_batch_grad, atol=ATOL, rtol=RTOL
        )
        assert torch.allclose(
            gsnr_from_sum_grad_squared, gsnr_from_batch_grad, rtol=RTOL, atol=ATOL
        ), "GSNRs from sum_grad_squared and batch_grad do not match"

        # sanity check 3: Both mean GSNRs match
        mean_gsnr_from_sum_grad_squared = gsnr_from_sum_grad_squared.mean()
        mean_gsnr_from_batch_grad = gsnr_from_batch_grad.mean()
        assert torch.allclose(
            gsnr_from_sum_grad_squared, gsnr_from_batch_grad, rtol=RTOL, atol=ATOL
        ), (
            "Mean GSNRs from sum_grad_squared and batch_grad do not match:"
            + f" {mean_gsnr_from_sum_grad_squared:.4f} vs."
            + f" {mean_gsnr_from_batch_grad:.4f}"
        )
