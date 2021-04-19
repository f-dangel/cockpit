"""Quantity for Takeuchi Information Criterion (TIC)."""

from backpack import extensions

from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook_SumGradSquared


class TIC(SingleStepQuantity):
    """Base class for different Takeuchi Information Criterion approximations.

    Note: Takeuchi Information criterion (TIC) rediscovered by

        - Thomas, V., et al.
          On the interplay between noise and curvature and its effect on
          optimization and generalization (2019).
          https://arxiv.org/abs/1906.07774
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
        """Initialization sets the tracking schedule & creates the output dict.

        Note:
            The curvature options ``"diag_h"`` and ``"diag_ggn_exact"`` are more
            expensive than ``"diag_ggn_mc"``, but more precise. For a classification
            task with ``C`` classes, the former require that ``C`` times more
            information be backpropagated through the computation graph.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
            curvature (str): Which diagonal curvature approximation should be used.
                Options are ``"diag_h"``, ``"diag_ggn_exact"``, ``"diag_ggn_mc"``.
            epsilon (float): Stabilization constant. Defaults to ``1e-7``.
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
            ext.append(extensions.BatchGrad())
            try:
                ext.append(self.extensions_from_str[self._curvature]())
            except KeyError as e:
                available = list(self.extensions_from_str.keys())
                raise KeyError(f"{str(e)}. Available: {available}")

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


class TICDiag(TIC):
    """Quantity class for tracking the TIC using diagonal curvature approximation.

    The diagonal curvature approximation provide cheap inversion.

    Note: Takeuchi Information criterion (TIC) rediscovered by

        - Thomas, V., et al.
          On the interplay between noise and curvature and its effect on
          optimization and generalization (2019).
          https://arxiv.org/abs/1906.07774
    """

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
    """Quantity class for the TIC using the trace of curvature and gradient covariance.

    Note: Takeuchi Information criterion (TIC) rediscovered by

        - Thomas, V., et al.
          On the interplay between noise and curvature and its effect on
          optimization and generalization (2019).
          https://arxiv.org/abs/1906.07774
    """

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
