"""Class for tracking the Mean Gradient Signal to Noise Ration (GSNR)."""

import torch

from backboard.quantities.quantity import SingleStepQuantity
from backboard.quantities.utils_quantities import (
    has_nans,
    has_negative,
    has_zeros,
    report_nonclose_values,
)
from backpack import extensions

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

    def __init__(
        self,
        track_interval=1,
        track_offset=0,
        epsilon=1e-5,
        use_double=False,
        verbose=False,
        check=False,
        track_schedule=None,
    ):
        """Initialize.

        Args:
            track_interval (int): Tracking rate.
            epsilon (float): Stabilization constant. Defaults to 0.0.
            use_double (bool): Whether to use doubles in computation. Defaults
                to ``False``.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
            check (bool): If True, this quantity will be computed via two different
                ways and compared. Defaults to ``False``.
        """
        super().__init__(
            track_interval=track_interval,
            track_offset=track_offset,
            verbose=verbose,
            track_schedule=track_schedule,
        )
        self._epsilon = epsilon
        self._use_double = use_double
        self._check = check

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        if self.is_active(global_step):
            ext = [extensions.SumGradSquared()]

            if self._check:
                ext.append(extensions.BatchGrad())

        else:
            ext = []

        return ext

    def compute(self, global_step, params, batch_loss):
        """Track the mean GSNR.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            mean_gsnr = self._compute(params, batch_loss).item()

            if self._verbose:
                print(f"[Step {global_step}] MeanGSNR: {mean_gsnr:.4f}")

            self.output[global_step]["mean_gsnr"] = mean_gsnr

            if self._check:
                self.__run_check(params, batch_loss)

    def _compute(self, params, batch_loss):
        """Return maximum θ for which the norm test would pass.

        The norm test is defined by Equation (3.9) in byrd2012sample.

        Args:
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        return self._compute_gsnr(params, batch_loss).mean()

    def _compute_gsnr(self, params, batch_loss):
        """Compute gradient signal-to-noise ratio.

        Args:
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self._use_double:
            grad_squared = self._fetch_grad(params, aggregate=True).double() ** 2
            sum_grad_squared = self._fetch_sum_grad_squared(
                params, aggregate=True
            ).double()
        else:
            grad_squared = self._fetch_grad(params, aggregate=True) ** 2
            sum_grad_squared = self._fetch_sum_grad_squared(params, aggregate=True)

        batch_size = self._fetch_batch_size_hotfix(batch_loss)

        return grad_squared / (
            batch_size * sum_grad_squared - grad_squared + self._epsilon
        )

    def __run_check(self, params, batch_loss):
        """Check if variance is non-negative and hence GSNR is not NaN."""

        def _compute_gsnr_from_batch_grad(params):
            """Gradient signal-to-noise ratio.

            Implement equation (25) in liu2020understanding, recursively defined via
            the prose between Equation (1) and (2).
            """
            batch_grad = self._fetch_batch_grad(params, aggregate=True)

            if self._use_double:
                batch_grad = batch_grad.double()

            batch_size = self._fetch_batch_size_hotfix(batch_loss)

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

        gsnr_from_sum_grad_squared = self._compute_gsnr(params, batch_loss)
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
