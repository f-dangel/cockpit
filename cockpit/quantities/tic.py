"""Quantity for Takeuchi Information Criterion (TIC)."""

import torch
from backpack import extensions
from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_quantities import (
    has_negative,
    has_zeros,
    report_nonclose_values,
)
from cockpit.quantities.utils_transforms import BatchGradTransforms_SumGradSquared

# TODO Move to tests
ATOL = 1e-5
RTOL = 5e-4


class TIC(SingleStepQuantity):
    """Base class for different Takeuchi Information Criterion approximations.

    Takeuchi Information criterion (TIC) rediscovered by thomas2019interplay.

    Link:
        - https://arxiv.org/pdf/1906.07774.pdf
    """

    extensions_from_str = {
        "diag_h": extensions.DiagHessian,
        "diag_ggn_exact": extensions.DiagGGNExact,
        "diag_ggn_mc": extensions.DiagGGNMC,
    }

    def __init__(
        self,
        track_schedule,
        verbose=False,
        curvature="diag_h",
        epsilon=1e-7,
    ):
        """Initialize TIC quantity.

        Note:
            The curvature options "diag_h" and "diag_ggn_exact" are more expensive than
            "diag_ggn_mc", but more precise. For a classification task with ``C``
            classes, the former require that ``C`` times more information be backpropa-
            gated through the computation graph.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
            curvature (string): Which diagonal curvature approximation should be used.
                Options are "diag_h", "diag_ggn_exact", "diag_ggn_mc".
            epsilon (float): Stabilization constant. Defaults to 1e-7.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._curvature = curvature
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
            try:
                ext.append(self.extensions_from_str[self._curvature]())
            except KeyError as e:
                available = list(self.extensions_from_str.keys())
                raise KeyError(f"{str(e)}. Available: {available}")

            ext.append(BatchGradTransforms_SumGradSquared())

        return ext


class TICDiag(TIC):
    """TIC with diagonal curvature approximation for cheap inversion."""

    def _compute(self, global_step, params, batch_loss):
        """Compute the TICDiag using a diagonal curvature approximation.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        sum_grad_squared = self._fetch_sum_grad_squared_via_batch_grad_transforms(
            params, aggregate=True
        )
        curvature = self._fetch_diag_curvature(params, self._curvature, aggregate=True)
        batch_size = get_batch_size(global_step)

        return (
            (batch_size * sum_grad_squared / (curvature + self._epsilon)).sum().item()
        )

    # TODO Move to tests
    def __run_check(self, global_step, params, batch_loss):
        """Run sanity checks for TICDiag."""

        def _compute_tic_with_batch_grad(params):
            """TICDiag."""
            batch_grad = self._fetch_batch_grad(params, aggregate=True)
            curvature = self._fetch_diag_curvature(
                params, self._curvature, aggregate=True
            )

            curv_stable = curvature + self._epsilon
            if has_zeros(curv_stable):
                raise ValueError("Diagonal curvature + ε has zeros.")
            if has_negative(curv_stable):
                raise ValueError("Diagonal curvature + ε has negative entries.")

            batch_size = get_batch_size(global_step)

            return torch.einsum("j,nj->", 1 / curv_stable, batch_size * batch_grad ** 2)

        # sanity check 1: Both TICDiags match
        tic_from_sgs = self._compute(global_step, params, batch_loss)
        tic_from_batch_grad = _compute_tic_with_batch_grad(params)

        report_nonclose_values(tic_from_sgs, tic_from_batch_grad, atol=ATOL, rtol=RTOL)
        assert torch.allclose(
            tic_from_sgs, tic_from_batch_grad, atol=ATOL, rtol=RTOL
        ), "TICDiags from sum_grad_squared and batch_grad do not match"


class TICTrace(TIC):
    """TIC approximation using the trace of curvature and gradient covariance."""

    def _compute(self, global_step, params, batch_loss):
        """Compute the TICTrace using a trace approximation.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        sum_grad_squared = self._fetch_sum_grad_squared_via_batch_grad_transforms(
            params, aggregate=True
        )
        curvature = self._fetch_diag_curvature(params, self._curvature, aggregate=True)
        batch_size = get_batch_size(global_step)

        return (
            batch_size * sum_grad_squared.sum() / (curvature.sum() + self._epsilon)
        ).item()

    # TODO Move to tests
    def __run_check(self, global_step, params, batch_loss):
        """Run sanity checks for TICTrace."""

        def _compute_tic_with_batch_grad(params):
            """TICTrace."""
            batch_grad = self._fetch_batch_grad(params, aggregate=True)
            curvature = self._fetch_diag_curvature(
                params, self._curvature, aggregate=True
            )

            curv_trace_stable = curvature.sum() + self._epsilon
            if has_zeros(curv_trace_stable):
                raise ValueError("Curvature trace + ε has zeros.")
            if has_negative(curv_trace_stable):
                raise ValueError("Curvature trace + ε has negative entries.")

            batch_size = get_batch_size(global_step)

            return batch_size * (batch_grad ** 2).sum() / curv_trace_stable

        # sanity check 1: Both TICTraces match
        tic_from_sgs = self._compute(global_step, params, batch_loss)
        tic_from_batch_grad = _compute_tic_with_batch_grad(params)

        report_nonclose_values(tic_from_sgs, tic_from_batch_grad, atol=ATOL, rtol=RTOL)
        assert torch.allclose(
            tic_from_sgs, tic_from_batch_grad, atol=ATOL, rtol=RTOL
        ), "TICTraces from sum_grad_squared and batch_grad do not match"
