"""Compare ``EarlyStopping`` quantity with ``torch.autograd``."""

import pytest
import torch

from cockpit.context import get_batch_size, get_individual_losses
from cockpit.quantities import EarlyStopping
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


class AutogradEarlyStopping(EarlyStopping):
    """``torch.autograd`` implementation of ``EarlyStopping``."""

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
        """Evaluate the early stopping criterion.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Early stopping criterion.
        """
        grad_squared = torch.cat([p.grad.flatten() for p in params]) ** 2

        losses = get_individual_losses(global_step)
        diag_variance = autograd_diagonal_variance(losses, params, concat=True)

        B = get_batch_size(global_step)

        return 1 - B * (grad_squared / (diag_variance + self._epsilon)).mean().item()


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_early_stopping(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of EarlyStopping.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    rtol, atol = 5e-3, 1e-5

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(
        problem, (EarlyStopping, AutogradEarlyStopping), q_kwargs, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("problem", ADAM_PROBLEMS, ids=ADAM_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_early_stopping_no_adam(problem, independent_runs, q_kwargs):
    """Verify Adam is not supported by EarlyStopping criterion.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    with pytest.raises(ValueError):
        test_early_stopping(problem, independent_runs, q_kwargs)
