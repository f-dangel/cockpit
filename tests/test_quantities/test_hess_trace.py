"""Compare ``HessTrace`` quantity with ``torch.autograd``."""

import pytest

from cockpit.quantities import HessTrace
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
)
from tests.test_quantities.utils import autograd_diag_hessian, get_compare_fn


class AutogradHessTrace(HessTrace):
    """``torch.autograd`` implementation of ``HessTrace``."""

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
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        return [
            diag_h.sum().item() for diag_h in autograd_diag_hessian(batch_loss, params)
        ]


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
def test_hess_trace(problem, independent_runs):
    """Compare BackPACK and ``torch.autograd`` implementation of Hessian trace.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (HessTrace, AutogradHessTrace), schedule)
