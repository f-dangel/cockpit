"""Quantity for Takeuchi Information Criterion (TIC)."""

from backpack import extensions

from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_SumGradSquared


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

        Raises:
            KeyError: If curvature string has unknown associated extension.

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

        Returns:
            float: TIC computed using a diagonal curvature approximation.
        """
        sum_grad_squared = self._fetch_sum_grad_squared_via_batch_grad_transforms(
            params, aggregate=True
        )
        curvature = self._fetch_diag_curvature(params, self._curvature, aggregate=True)
        batch_size = get_batch_size(global_step)

        return (
            (batch_size * sum_grad_squared / (curvature + self._epsilon)).sum().item()
        )


class TICTrace(TIC):
    """TIC approximation using the trace of curvature and gradient covariance."""

    def _compute(self, global_step, params, batch_loss):
        """Compute the TICTrace using a trace approximation.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: TIC computed using a trace approximation.
        """
        sum_grad_squared = self._fetch_sum_grad_squared_via_batch_grad_transforms(
            params, aggregate=True
        )
        curvature = self._fetch_diag_curvature(params, self._curvature, aggregate=True)
        batch_size = get_batch_size(global_step)

        return (
            batch_size * sum_grad_squared.sum() / (curvature.sum() + self._epsilon)
        ).item()
