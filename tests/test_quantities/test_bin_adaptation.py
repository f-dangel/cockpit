"""Test bin adaptation policies."""

import pytest

from cockpit.context import get_individual_losses
from cockpit.quantities import GradHist1d
from cockpit.quantities.bin_adaptation import GradAbsMax
from tests.test_quantities.settings import (
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.test_grad_hist import AutogradGradHist1d
from tests.test_quantities.utils import (
    autograd_individual_gradients,
    run_harness_get_output,
)
from tests.utils.check import compare_outputs


class AutogradGradAbsMax(GradAbsMax):
    """Autograd implementation of ``GradAbsMax`` policy."""

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return self.should_compute(global_step)

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def _compute(self, global_step, params, batch_loss, range):
        """Evaluate new histogram limits.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range ((float, float)): Current bin limits.

        Returns:
            (float, float): New bin ranges.
        """
        individual_losses = get_individual_losses(global_step)
        individual_gradients = autograd_individual_gradients(
            individual_losses, params, concat=True
        )
        abs_max = individual_gradients.abs().max().item()

        end = (1.0 + self._padding) * max(self._min_size / 2, abs_max)
        start = -end

        return start, end


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_grad_hist1d_adapted(problem, q_kwargs):
    """Compare the 1d histogram with bin adaptation versus autograd.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """

    def adapt_schedule(global_step):
        return global_step in [1, 2]

    q1 = GradHist1d(**q_kwargs, adapt=GradAbsMax(adapt_schedule, verbose=True))
    output1 = run_harness_get_output(problem, [q1])[0]

    q2 = AutogradGradHist1d(
        **q_kwargs, adapt=AutogradGradAbsMax(adapt_schedule, verbose=True)
    )
    output2 = run_harness_get_output(problem, [q2])[0]

    compare_outputs(output1, output2)
