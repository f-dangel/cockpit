"""Adaptation policies to dynamically update histogram bins."""

from backpack import extensions

from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_hists import transform_grad_batch_abs_max


class BinAdaptation(SingleStepQuantity):
    """Base class for policies to adapt the bin limits of histogram quantities."""

    def compute(self, global_step, params, batch_loss, range):
        """Evaluate the new histogram limits.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range ((float, float)): Current bin limits.

        Returns:
            (float, float): New bin ranges.
        """
        start, end = self._compute(global_step, params, batch_loss, range)

        if self._verbose:
            print(
                f"{self._verbose_prefix(global_step)}:" + f" New limits: {start}, {end}"
            )

        return start, end

    def _compute(self, global_step, params, batch_loss, range):
        """Evaluate new histogram limits.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range ((float, float)): Current bin limits.

        Returns:
            (float, float): New bin ranges.
        """
        raise NotImplementedError


class GradAbsMax(BinAdaptation):
    """Bin adaptation policy using the absolute maximum of individual gradient elements.

    Updates bins to ``[-m; m]`` where ``m`` is the maximum of the individual gradient
    element absolute values. Optionally adds padding.
    """

    def __init__(self, track_schedule, verbose=False, padding=0.0, min_size=1e-6):
        """Initialize the individual gradient range bin adaptation policy.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose
            padding (float, optional): Relative padding added to the gradient range.
                Defaults to ``0.0``.
            min_size (float, optional): Minimum range size. Default: ``1e-6``.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._padding = padding
        self._min_size = min_size

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.should_compute(global_step):
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={"grad_batch_abs_max": transform_grad_batch_abs_max}
                )
            )

        return ext

    def _compute(self, global_step, params, batch_loss, range):
        """Evaluate new histogram limits.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range ((float, float)): Current bin limits.

        Returns:
            (float, float): New bin ranges.
        """
        abs_max = max(p.grad_batch_transforms["grad_batch_abs_max"] for p in params)

        end = (1.0 + self._padding) * max(self._min_size / 2, abs_max)
        start = -end

        return start, end
