"""Compare ``HessTrace`` quantity with ``torch.autograd``."""

import pytest

from cockpit.quantities import HessTrace
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import PROBLEMS, PROBLEMS_IDS
from tests.test_quantities.utils import (
    autograd_diag_hessian,
    compare_quantities_separate_runs,
    compare_quantities_single_run,
)


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
def test_hess_trace_single_run(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of Hessian trace.

    Both quantities run simultaneously in the same cockpit.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)

    compare_quantities_single_run(problem, (HessTrace, AutogradHessTrace), schedule)


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_hess_trace_separate_runs(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of Hessian trace.

    Both quantities run in separate cockpits.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)

    compare_quantities_separate_runs(problem, (HessTrace, AutogradHessTrace), schedule)
