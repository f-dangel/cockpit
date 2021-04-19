"""Compare ``OrthoTest`` quantity with ``torch.autograd``."""

import pytest
import torch

from cockpit.context import get_batch_size, get_individual_losses
from cockpit.quantities import OrthoTest
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.utils import autograd_individual_gradients, get_compare_fn


class AutogradOrthoTest(OrthoTest):
    """``torch.autograd`` implementation of ``OrthoTest``."""

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
        """Evaluate the norm test.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            foat: Result of the norm test.
        """
        losses = get_individual_losses(global_step)
        individual_gradients_flat = autograd_individual_gradients(
            losses, params, concat=True
        )
        D_axis = 1
        individual_l2_norms_squared = (individual_gradients_flat ** 2).sum(D_axis)

        grad = torch.cat([p.grad.flatten() for p in params])
        grad_norm = grad.norm()

        projections = torch.einsum("ni,i->n", individual_gradients_flat, grad)

        batch_size = get_batch_size(global_step)

        return (
            (
                1
                / (batch_size * (batch_size - 1))
                * (
                    individual_l2_norms_squared / grad_norm ** 2
                    - (projections ** 2) / grad_norm ** 4
                ).sum()
            )
            .sqrt()
            .item()
        )


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_ortho_test(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of OrthoTest.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (OrthoTest, AutogradOrthoTest), q_kwargs)
