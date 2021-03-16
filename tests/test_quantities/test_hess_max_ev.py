"""Compare ``HessMaxEV`` quantity with ``torch.autograd``."""

import warnings

import pytest

from cockpit.quantities import HessMaxEV
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import PROBLEMS, PROBLEMS_IDS
from tests.test_quantities.utils import (
    autograd_hessian_maximum_eigenvalue,
    compare_quantities_separate_runs,
    compare_quantities_single_run,
)


class AutogradHessMaxEV(HessMaxEV):
    """``torch.autograd`` implementation of ``HessMaxEV``.

    Requires storing the full Hessian in memory and can hence only be applied to small
    networks.
    """

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the maximum mini-batch loss Hessian eigenvalue.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        self._maybe_warn_dimension(sum(p.numel() for p in params))

        return autograd_hessian_maximum_eigenvalue(batch_loss, params).item()

    @staticmethod
    def _maybe_warn_dimension(dim):
        """Warn user if the Hessian is large."""
        MAX_DIM = 1000

        if dim >= MAX_DIM:
            warnings.warn(f"Computing Hessians of size ({dim}, {dim}) is expensive")


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_hess_max_ev_single_run(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of Hessian max eigenvalue.

    Both quantities run simultaneously in the same cockpit.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    atol, rtol = 1e-4, 1e-2

    compare_quantities_single_run(
        problem, (HessMaxEV, AutogradHessMaxEV), schedule, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_hess_max_ev_separate_runs(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of Hessian max eigenvalue.

    Both quantities run in separate cockpits.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    atol, rtol = 1e-4, 1e-2

    compare_quantities_separate_runs(
        problem, (HessMaxEV, AutogradHessMaxEV), schedule, rtol=rtol, atol=atol
    )
