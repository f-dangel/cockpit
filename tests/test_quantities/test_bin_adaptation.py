"""Test bin adaptation policies."""

import pytest

from cockpit.context import get_individual_losses
from cockpit.quantities import GradHist1d, GradHist2d
from cockpit.quantities.bin_adaptation import GradAbsMax, ParamAbsMax
from tests.test_quantities.settings import (
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.test_grad_hist import AutogradGradHist1d, AutogradGradHist2d
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

    def extension_hooks(self, global_step):
        """Return list of BackPACK extension hooks required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            [callable]: List of required BackPACK extension hooks for the current
                iteration.
        """
        return []

    def _get_abs_max(self, global_step, params, batch_loss, range):
        """Compute the maximum absolute value of individual gradient elements.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            range (float, float): Current bin limits.

        Returns:
            float: Maximum absolute value of individual gradients.
        """
        individual_losses = get_individual_losses(global_step)
        individual_gradients = autograd_individual_gradients(
            individual_losses, params, concat=True
        )
        return individual_gradients.abs().max().item()


class AutogradParamAbsMax(ParamAbsMax):
    """Autograd implementation of ``ParamAbsMax`` policy.

    The parent class already only uses ``autograd``.
    """

    pass


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


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_grad_hist2d_adapted(problem, q_kwargs):
    """Compare the 2d histogram with bin adaptation versus autograd.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """

    def adapt_schedule(global_step):
        return global_step in [1, 2]

    q1 = GradHist2d(
        **q_kwargs,
        adapt=(
            GradAbsMax(adapt_schedule, verbose=True),
            ParamAbsMax(adapt_schedule, verbose=True),
        ),
    )
    output1 = run_harness_get_output(problem, [q1])[0]

    q2 = AutogradGradHist2d(
        **q_kwargs,
        adapt=(
            AutogradGradAbsMax(adapt_schedule, verbose=True),
            AutogradParamAbsMax(adapt_schedule, verbose=True),
        ),
    )
    output2 = run_harness_get_output(problem, [q2])[0]

    compare_outputs(output1, output2)
