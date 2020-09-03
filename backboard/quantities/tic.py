"""Quantity for Takeuchi Information Criterion (TIC)."""

import torch

from backboard.quantities.quantity import Quantity
from backpack import extensions
from tests.utils import has_negative, has_zeros, report_nonclose_values

# TODO Read up where to use mini-batch and dataset size
# TODO Maybe use the cheaper approximation: fraction of traces

ATOL = 1e-7
RTOL = 1e-5


class TakeuchiInformationCriterion(Quantity):
    """Takeuchi Information criterion (TIC) rediscovered by thomas2019interplay.

    Uses a diagonal curvature approximation.

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
        track_interval,
        curvature="diag_h",
        epsilon=0.0,
        use_double=False,
        verbose=False,
        check=False,
    ):
        """Initialize TIC quantity.

        Args:
            track_interval (int): Tracking rate.
            curvature (string): Which diagonal curvature approximation should be used.
                Options are "diag_h", "diag_ggn_exact", "diag_ggn_mc".
            epsilon (float): Stabilization constant. Defaults to 0.0.
            use_double (bool): Whether to use doubles in computation. Defaults
                to ``False``.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
            check (bool): If True, this quantity will be computed via two different
                ways and compared. Defaults to ``False``.
        """
        super().__init__(track_interval)
        self._curvature = curvature
        self._epsilon = epsilon
        self._use_double = use_double
        self._verbose = verbose
        self._check = check

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        if global_step % self._track_interval == 0:
            try:
                ext = [self.extensions_from_str[self._curvature]()]
            except KeyError as e:
                available = list(self.extensions_from_str.keys())
                raise KeyError(f"{str(e)}. Available: {available}")

            if self._check:
                ext.append(extensions.BatchGrad())

            ext.append(extensions.SumGradSquared())

        else:
            ext = []

        return ext

    def compute(self, global_step, params, batch_loss):
        """Compute practical version of norm test (byrd2012sample).

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        """
        if global_step % self._track_interval == 0:
            tic = self._compute(params, batch_loss)
            self.output[global_step]["tic"] = [tic.item()]

            if self._check:
                self.__run_check(params, batch_loss)

        else:
            pass

    def _compute(self, params, batch_loss):
        """Return TIC value.

        Args:
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        tic = self._compute_tic(params, batch_loss)

        if self._verbose:
            print(f"Takeuchi Information Criterion TIC={tic:.4f}")

        return tic

    def _compute_tic(self, params, batch_loss):
        """Compute the TIC using a diagonal curvature approximation.

        Args:
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        sum_grad_squared = self._fetch_sum_grad_squared(params, aggregate=True)
        curvature = self._fetch_diag_curvature(params, self._curvature, aggregate=True)

        if self._use_double:
            sum_grad_squared = sum_grad_squared.double()
            curvature = curvature.double()

        return (sum_grad_squared / (curvature + self._epsilon)).sum()

    def __run_check(self, params, batch_loss):
        """Run sanity checks for TIC."""

        def _compute_tic_with_batch_grad(params):
            """TIC."""
            batch_grad = self._fetch_batch_grad(params, aggregate=True)
            curvature = self._fetch_diag_curvature(
                params, self._curvature, aggregate=True
            )

            if self._use_double:
                batch_grad = batch_grad.double()
                curvature = curvature.double()

            curv_stable = curvature + self._epsilon
            if has_zeros(curv_stable):
                raise ValueError("Diagonal curvature + ε has zeros.")
            if has_negative(curv_stable):
                raise ValueError("Diagonal curvature + ε has negative entries.")

            return torch.einsum("j,nj->", 1 / curv_stable, batch_grad ** 2)

        # sanity check 1: Both TICs match
        tic_from_sgs = self._compute(params, batch_loss)
        tic_from_batch_grad = _compute_tic_with_batch_grad(params)

        report_nonclose_values(tic_from_sgs, tic_from_batch_grad, atol=ATOL, rtol=RTOL)
        assert torch.allclose(
            tic_from_sgs, tic_from_batch_grad, atol=ATOL, rtol=RTOL
        ), "TICs from sum_grad_squared and batch_grad do not match"
