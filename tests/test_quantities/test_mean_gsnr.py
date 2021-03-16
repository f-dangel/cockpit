"""Compare ``MeanGSNR`` quantity with ``torch.autograd``."""

import pytest
import torch

from cockpit.context import get_individual_losses
from cockpit.quantities import MeanGSNR
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import PROBLEMS, PROBLEMS_IDS
from tests.test_quantities.utils import (
    autograd_individual_gradients,
    compare_quantities_separate_runs,
    compare_quantities_single_run,
)


class AutogradMeanGSNR(MeanGSNR):
    """``torch.autograd`` implementation of ``MeanGSNR``."""

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
        return True

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the TIC approximating the Hessian by its diagonal.

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

        grad_squared = torch.cat([p.grad.flatten() for p in params]) ** 2

        N_axis = 0
        second_moment_flat = (individual_gradients_flat ** 2).mean(N_axis)

        gsnr = grad_squared / (second_moment_flat - grad_squared + self._epsilon)

        return gsnr.mean().item()


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_mean_gsnr_single_run(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of MeanGSNR.

    Both quantities run simultaneously in the same cockpit.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    rtol, atol = 5e-3, 1e-5

    compare_quantities_single_run(
        problem, (MeanGSNR, AutogradMeanGSNR), schedule, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_mean_gsnr_separate_runs(problem):
    """Compare BackPACK and ``torch.autograd`` implementation of MeanGSNR.

    Both quantities run in separate cockpits.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    rtol, atol = 5e-3, 1e-5

    compare_quantities_separate_runs(
        problem, (MeanGSNR, AutogradMeanGSNR), schedule, rtol=rtol, atol=atol
    )
