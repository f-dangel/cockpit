"""Class for tracking the Trace of the Hessian or an approximation thereof."""

from backpack import extensions
from cockpit.quantities.quantity import SingleStepQuantity


class Trace(SingleStepQuantity):
    """Trace Quantitiy Class."""

    extensions_from_str = {
        "diag_h": extensions.DiagHessian,
        "diag_ggn_exact": extensions.DiagGGNExact,
        "diag_ggn_mc": extensions.DiagGGNMC,
    }

    def __init__(self, track_schedule, verbose=False, curvature="diag_h"):
        """Initialize the trace quantity.

        Crucially, it creates the output dictionary, that is meant to store all
        values that should be stored.

        Note:
            The curvature options "diag_h" and "diag_ggn_exact" are more expensive than
            "diag_ggn_mc", but more precise. For a classification task with ``C``
            classes, the former require that ``C`` times more information be backpropa-
            gated through the computation graph.

        Args:
            curvature (string): Which diagonal curvature approximation should be used.
                Options are "diag_h", "diag_ggn_exact", "diag_ggn_mc".
        """
        super().__init__(track_schedule, verbose=verbose)

        self._curvature = curvature

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.is_active(global_step):
            try:
                ext.append(self.extensions_from_str[self._curvature]())
            except KeyError as e:
                available = list(self.extensions_from_str.keys())
                raise KeyError(f"{str(e)}. Available: {available}")

        return ext

    # TODO Rewrite to use parent class track method
    def track(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            trace = [
                diag_c.sum().item()
                for diag_c in self._fetch_diag_curvature(
                    params, self._curvature, aggregate=False
                )
            ]

            if self._verbose:
                print(f"[Step {global_step}] Trace: {sum(trace):.4f}")

            self.output[global_step]["trace"] = trace
