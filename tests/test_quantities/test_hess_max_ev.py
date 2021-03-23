"""Compare ``HessMaxEV`` quantity with ``torch.autograd``."""

import warnings

import pytest

from cockpit.quantities import HessMaxEV
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.utils import (
    autograd_hessian_maximum_eigenvalue,
    get_compare_fn,
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

        Returns:
            float: Maximum Hessian eigenvalue (of the mini-batch loss).
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
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_hess_max_ev(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of Hessian max eigenvalue.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    atol, rtol = 1e-4, 1e-2

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (HessMaxEV, AutogradHessMaxEV), q_kwargs, rtol=rtol, atol=atol)
