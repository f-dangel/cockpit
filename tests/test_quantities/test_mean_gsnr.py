"""Compare ``MeanGSNR`` quantity with ``torch.autograd``."""

import pytest
import torch

from cockpit.context import get_individual_losses
from cockpit.quantities import MeanGSNR
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.utils import autograd_individual_gradients, get_compare_fn


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
        """Evaluate the MeanGSNR.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Mean GSNR of the current iteration.
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
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_mean_gsnr(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of MeanGSNR.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    rtol, atol = 5e-3, 1e-5

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (MeanGSNR, AutogradMeanGSNR), q_kwargs, rtol=rtol, atol=atol)
