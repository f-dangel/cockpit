"""Compare ``TICDiag`` and ``TICTrace`` quantities with ``torch.autograd``."""

import pytest

from cockpit.context import get_individual_losses
from cockpit.quantities import TICDiag, TICTrace
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.utils import (
    autograd_diag_hessian,
    autograd_individual_gradients,
    get_compare_fn,
)


class AutogradTICDiag(TICDiag):
    """``torch.autograd`` implementation of ``TICDiag``."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def extension_hooks(self, global_step):
        """Return list of BackPACK extension hooks required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            [callable]: List of required BackPACK extension hooks for the current
                iteration.
        """
        return []

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return self.should_compute(global_step)

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the TIC approximating the Hessian by its diagonal.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Raises:
            NotImplementedError: If curvature is not ``diag_h``.

        Returns:
            float: TIC when approximation the Hessian with its diagonal.
        """
        losses = get_individual_losses(global_step)
        individual_gradients_flat = autograd_individual_gradients(
            losses, params, concat=True
        )

        if self._curvature == "diag_h":
            diag_curvature_flat = autograd_diag_hessian(batch_loss, params, concat=True)
        else:
            raise NotImplementedError("Only Hessian diagonal is implemented")

        N_axis = 0
        second_moment_flat = (individual_gradients_flat ** 2).mean(N_axis)

        return (second_moment_flat / (diag_curvature_flat + self._epsilon)).sum().item()


class AutogradTICTrace(TICTrace):
    """``torch.autograd`` implementation of ``TICTrace``."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return self.should_compute(global_step)

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the TIC approximation proposed in thomas2020interplay.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Raises:
            NotImplementedError: If curvature is not ``diag_h``.

        Returns:
            float: TIC using a trace approximation.
        """
        losses = get_individual_losses(global_step)
        individual_gradients_flat = autograd_individual_gradients(
            losses, params, concat=True
        )

        if self._curvature == "diag_h":
            curvature_trace = autograd_diag_hessian(
                batch_loss, params, concat=True
            ).sum()
        else:
            raise NotImplementedError("Only Hessian trace is implemented")

        N_axis = 0
        mean_squared_l2_norm = (individual_gradients_flat ** 2).mean(N_axis).sum()

        return (mean_squared_l2_norm / (curvature_trace + self._epsilon)).item()


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_tic_diag(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of TICDiag.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (TICDiag, AutogradTICDiag), q_kwargs)


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_tic_trace(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of TICTrace.

    Both quantities run simultaneously in the same cockpit.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (TICTrace, AutogradTICTrace), q_kwargs)
