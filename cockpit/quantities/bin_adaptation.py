"""Adaptation policies to dynamically update histogram bins."""

from backpack.extensions import BatchGrad

from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_hists import transform_grad_batch_abs_max
from cockpit.quantities.utils_transforms import BatchGradTransformsHook


class BinAdaptation(SingleStepQuantity):
    """Base class for policies to adapt the bin limits of histogram quantities."""

    def compute(self, global_step, params, batch_loss, range):
        """Evaluate the new histogram limits.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range (float, float): Current bin limits.

        Returns:
            (float, float): New bin ranges.
        """
        start, end = self._compute(global_step, params, batch_loss, range)

        if self._verbose:
            print(
                f"{self._verbose_prefix(global_step)}:"
                + f" New limits: {start:.5f}, {end:.5f}"
            )

        return start, end

    def _compute(self, global_step, params, batch_loss, range):
        """Evaluate new histogram limits.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range (float, float): Current bin limits.

        Returns: # noqa: DAR202
            (float, float): New bin ranges.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError


class NoAdaptation(BinAdaptation):
    """Leave histogram bin ranges unaffected."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def __init__(self, verbose=False):
        """Never adapt the bins by using a schedule that is never triggered.

        Args:
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """

        def track_schedule(global_step):
            """No scheduled computations."""
            return False

        super().__init__(track_schedule, verbose=verbose)


class _AbsMax(BinAdaptation):
    """Base class for bin adaptation based on a maximum absolute value.

    Updates bins to ``[-m; m]`` where ``m`` is the maximum absolute value. Optionally
    adds padding.
    """

    def __init__(self, track_schedule, verbose=False, padding=0.0, min_size=1e-6):
        """Initialize the bin adaptation policy.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose
            padding (float, optional): Relative padding added to the maximum absolute
                value. Defaults to ``0.0``.
            min_size (float, optional): Minimum range size. Default: ``1e-6``.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._padding = padding
        self._min_size = min_size

    def _compute(self, global_step, params, batch_loss, range):
        """Evaluate new histogram limits.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range (float, float): Current bin limits.

        Returns:
            (float, float): New bin ranges.
        """
        abs_max = self._get_abs_max(global_step, params, batch_loss, range)

        end = (1.0 + self._padding) * max(self._min_size / 2, abs_max)
        start = -end

        return start, end

    def _get_abs_max(self, global_step, params, batch_loss, range):
        """Compute the maximum absolute value.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range (float, float): Current bin limits.

        Returns: # noqa: DAR202
            float: Maximum absolute value used to compute the new bin limits.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError


class GradAbsMax(_AbsMax):
    """Bin adaptation policy using the absolute maximum of individual gradient elements.

    Updates bins to ``[-m; m]`` where ``m`` is the maximum of the individual gradient
    element absolute values. Optionally adds padding.
    """

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
            hooks.append(
                BatchGradTransformsHook(
                    transforms={"grad_batch_abs_max": transform_grad_batch_abs_max}
                )
            )

        return hooks

    def _get_abs_max(self, global_step, params, batch_loss, range):
        """Compute the maximum absolute value of individual gradient elements.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range (float, float): Current bin limits.

        Returns:
            float: Maximum absolute value of individual gradients.
        """
        return max(p.grad_batch_transforms["grad_batch_abs_max"] for p in params)


class ParamAbsMax(_AbsMax):
    """Bin adaptation policy using the absolute maximum of parameer elements.

    Updates bins to ``[-m; m]`` where ``m`` is the maximum of the parameter element
    absolute values. Optionally adds padding.
    """

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def _get_abs_max(self, global_step, params, batch_loss, range):
        """Compute the maximum absolute value of parameter elements.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range (float, float): Current bin limits.

        Returns:
            float: Maximum absolute value of parameters.
        """
        return max(p.data.abs().max() for p in params).item()
