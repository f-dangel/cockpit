"""Compare ``InnerTest`` quantity with ``torch.autograd``."""

import pytest
import torch

from cockpit.context import get_batch_size, get_individual_losses
from cockpit.quantities import InnerTest
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import PROBLEMS, PROBLEMS_IDS
from tests.test_quantities.utils import (
    autograd_individual_gradients,
    compare_quantities_separate_runs,
    compare_quantities_single_run,
)


class AutogradInnerTest(InnerTest):
    """``torch.autograd`` implementation of ``InnerTest``."""

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
        """Evaluate the inner-product test.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        losses = get_individual_losses(global_step)
        individual_gradients_flat = autograd_individual_gradients(
            losses, params, concat=True
        )
        grad = torch.cat([p.grad.flatten() for p in params])

        projections = torch.einsum("ni,i->n", individual_gradients_flat, grad)
        grad_norm = grad.norm()

        N_axis = 0
        batch_size = get_batch_size(global_step)

        return (
            (
                1
                / (batch_size * (batch_size - 1))
                * ((projections ** 2).sum(N_axis) / grad_norm ** 4 - batch_size)
            )
            .sqrt()
            .item()
        )


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_inner_test_single_run(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of InnerTest.

    Both quantities run simultaneously in the same cockpit.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    rtol, atol = 5e-3, 1e-5

    compare_quantities_single_run(
        problem, (InnerTest, AutogradInnerTest), schedule, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_inner_test_separate_runs(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of InnerTest.

    Both quantities run in separate cockpits.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    rtol, atol = 5e-3, 1e-5

    compare_quantities_separate_runs(
        problem, (InnerTest, AutogradInnerTest), schedule, rtol=rtol, atol=atol
    )
