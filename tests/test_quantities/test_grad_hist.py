"""Compare ``GradHist{1,2}d`` quantities with ``torch.autograd``."""

import warnings

import numpy
import pytest

from cockpit.context import get_individual_losses
from cockpit.quantities import GradHist1d
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
)
from tests.test_quantities.utils import autograd_individual_gradients, get_compare_fn


class AutogradGradHist1d(GradHist1d):
    """``torch.autograd,numpy`` implementation of ``GradHist1d``."""

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
        return self.should_compute(global_step) or self._adapt_schedule(global_step)

    # TODO Rewrite to use parent class track method
    def track(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.should_compute(global_step):
            edges = self._get_current_bin_edges().cpu().numpy()

            losses = get_individual_losses(global_step)
            individual_gradients = autograd_individual_gradients(
                losses, params, concat=False
            )
            hist = sum(
                self._compute_histogram(igrad.detach().cpu().numpy())
                for igrad in individual_gradients
            )

            self.output[global_step]["hist_1d"] = hist.tolist()
            self.output[global_step]["edges"] = edges.tolist()

            if self._verbose:
                print(
                    f"[Step {global_step}] AutogradGradHist1d"
                    + f" edges 0,...,4: {edges[:5]}"
                )
                print(
                    f"[Step {global_step}] AutogradGradHist1d"
                    + f" counts 0,...,4: {hist[:5]}"
                )

        self._update_limits(global_step, params, batch_loss)

    def _compute_histogram(self, individual_gradients):
        """Compute bin counts of individual gradient elements."""
        hist, _ = numpy.histogram(
            individual_gradients, bins=self._bins, range=(self._xmin, self._xmax)
        )

        return hist

    def _update_limits(self, global_step, params, batch_loss):
        """Update limits for next histogram computation."""
        if self._adapt_schedule(global_step):
            pad_factor = 1.0 + self._pad

            losses = get_individual_losses(global_step)
            individual_gradients = autograd_individual_gradients(
                losses, params, concat=False
            )
            abs_max = float(
                pad_factor * max(igrad.abs().max() for igrad in individual_gradients)
            )

            if abs_max == 0.0:
                warnings.warn(
                    "Adaptive x limits are identical, using a small range instead."
                )
                epsilon = 1e-6
                abs_max += epsilon

            self._xmin, self._xmax = -abs_max, abs_max

            if self._verbose:
                print(
                    f"[Step {global_step}] AutogradGradHist1d"
                    + f" new limits: ({self._xmin:.4f}, {self._xmax:.4f})",
                )


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
def test_grad_hist(problem, independent_runs):
    """Compare BackPACK and ``torch.autograd`` implementation of GradHist.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (GradHist1d, AutogradGradHist1d), schedule)
