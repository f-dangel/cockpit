"""Compare ``CABS`` quantity with ``torch.autograd``."""

import pytest

from cockpit.context import get_individual_losses, get_optimizer
from cockpit.quantities import CABS
from cockpit.utils.optim import ComputeStep
from tests.test_quantities.adam_settings import ADAM_IDS, ADAM_PROBLEMS
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.utils import autograd_diagonal_variance, get_compare_fn


class AutogradCABS(CABS):
    """``torch.autograd`` implementation of ``CABS``."""

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
        """Evaluate the CABS criterion.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Evaluated CABS criterion.

        Raises:
            ValueError: If the optimizer differs from SGD with default arguments.
        """
        optimizer = get_optimizer(global_step)
        if not ComputeStep.is_sgd_default_kwargs(optimizer):
            raise ValueError("This criterion only supports zero-momentum SGD.")

        losses = get_individual_losses(global_step)
        batch_axis = 0
        trace_variance = autograd_diagonal_variance(
            losses, params, concat=True, unbiased=False
        ).sum(batch_axis)
        lr = self.get_lr(optimizer)

        return (lr * trace_variance / batch_loss).item()


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_cabs(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of CABS.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (CABS, AutogradCABS), q_kwargs)


@pytest.mark.parametrize("problem", ADAM_PROBLEMS, ids=ADAM_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_cabs_no_adam(problem, independent_runs, q_kwargs):
    """Verify Adam is not supported by CABS criterion.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    with pytest.raises(ValueError):
        test_cabs(problem, independent_runs, q_kwargs)
