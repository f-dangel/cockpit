"""Class for tracking the Trace of the Hessian or an approximation thereof."""

from backpack import extensions

from cockpit.quantities.quantity import SingleStepQuantity


class HessTrace(SingleStepQuantity):
    """Quantitiy Class tracking the trace of the Hessian during training."""

    extensions_from_str = {
        "diag_h": extensions.DiagHessian,
        "diag_ggn_exact": extensions.DiagGGNExact,
        "diag_ggn_mc": extensions.DiagGGNMC,
    }

    def __init__(self, track_schedule, verbose=False, curvature="diag_h"):
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
            curvature (string): Which diagonal curvature approximation should be used.
                Options are ``"diag_h"``, ``"diag_ggn_exact"``, ``"diag_ggn_mc"``.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._curvature = curvature

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

        return ext

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            list: Trace of the Hessian at the current point.
        """
        return [
            diag_c.sum().item()
            for diag_c in self._fetch_diag_curvature(
                params, self._curvature, aggregate=False
            )
        ]
