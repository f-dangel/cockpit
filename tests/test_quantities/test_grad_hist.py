"""Compare ``GradHist{1,2}d`` quantities with ``torch.autograd``."""

import warnings

import numpy
import pytest
import torch

from cockpit.context import get_individual_losses
from cockpit.quantities import GradHist1d, GradHist2d
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import (
    CPU_PROBLEMS,
    CPU_PROBLEMS_ID,
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

        return hist.astype(float)

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


class AutogradGradHist2d(GradHist2d):
    """``torch.autograd,numpy`` implementation of ``GradHist2d``."""

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
        """Compute the two-dimensional histogram at the current iteration.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.should_compute(global_step):
            self._compute_aggregated(global_step, params, batch_loss)

            if self._keep_individual:
                self._compute_individual(global_step, params, batch_loss)

        self._update_limits(global_step, params, batch_loss)

    def _compute_aggregated(self, global_step, params, batch_loss):
        """Aggregate histogram data over parameters and save to output."""
        x_edges, y_edges = self._get_current_bin_edges()

        losses = get_individual_losses(global_step)
        individual_gradients = autograd_individual_gradients(losses, params)

        hist = None

        for igrad, param in zip(individual_gradients, params):
            param_hist = self._compute_histogram(param, igrad)

            hist = param_hist if hist is None else hist + param_hist

        self.output[global_step]["hist_2d"] = hist.tolist()
        self.output[global_step]["x_edges"] = x_edges.cpu().numpy().tolist()
        self.output[global_step]["y_edges"] = y_edges.cpu().numpy().tolist()

        if self._verbose:
            print(
                f"[Step {global_step}] AutogradGradHist2d"
                + f" x_edges 0,...,4: {x_edges[:5]}"
            )
            print(
                f"[Step {global_step}] AutogradGradHist2d"
                + f" y_edges 0,...,4: {y_edges[:5]}"
            )
            print(
                f"[Step {global_step}] AutogradGradHist2d"
                + f" counts [0,...,4][0,...,4]: {hist[:5,:5]}"
            )
        print(hist.max())

    def _compute_histogram(self, param, individual_gradients):
        """Compute bin counts of individual gradient elements."""
        hist_bins = (self._xbins, self._ybins)
        hist_range = ((self._xmin, self._xmax), (self._ymin, self._ymax))

        batch_axis = 0
        batch_size = individual_gradients.shape[batch_axis]
        individual_gradients_clamped, param_clamped = self.__preprocess(
            individual_gradients.data, param.data
        )

        hist = None

        for b in range(batch_size):
            igrad = individual_gradients_clamped[b].flatten()

            hist_b, _, _ = numpy.histogram2d(
                igrad.detach().cpu().numpy(),
                param_clamped.flatten().detach().cpu().numpy(),
                bins=hist_bins,
                range=hist_range,
            )

            hist = hist_b if hist is None else hist + hist_b

        return hist.astype(float)

    def __preprocess(self, batch_grad, param):
        """Scale and clamp the data used for histograms."""
        # clip to interval, elements outside [xmin, xmax] would be ignored
        xmin, xmax = self._xmin, self._xmax
        ymin, ymax = self._ymin, self._ymax

        # PyTorch implementation has different comparison conventions
        if True:
            xedges, yedges = self._get_current_bin_edges()
            xbin_size, ybin_size = xedges[1] - xedges[0], yedges[1] - yedges[0]
            xepsilon, yepsilon = xbin_size / 2, ybin_size / 2

            xmin, xmax = xmin + xepsilon, xmax - xepsilon
            ymin, ymax = ymin + yepsilon, ymax - yepsilon
        batch_grad_clamped = torch.clamp(batch_grad, xmin, xmax)
        param_clamped = torch.clamp(param, ymin, ymax)

        return batch_grad_clamped, param_clamped

    def _update_limits(self, global_step, params, batch_loss):
        """Update limits for next histogram computation."""
        if self._adapt_schedule(global_step):
            self._update_x_limits(params, global_step)
            self._update_y_limits(params)

            if self._verbose:
                print(
                    f"[Step {global_step}] AutogradGradHist2d"
                    + f" new x limits: ({self._xmin:.4f}, {self._xmax:.4f})",
                )
                print(
                    f"[Step {global_step}] AutogradGradHist2d"
                    + f" new y limits: ({self._ymin:.4f}, {self._ymax:.4f})",
                )

    def _update_x_limits(self, params, global_step):
        """Update the histogram's x limits."""
        losses = get_individual_losses(global_step)
        individual_gradients = autograd_individual_gradients(losses, params)

        if self._adapt_policy == "abs_max":
            pad_factor = 1 + self._xpad
            abs_max = float(max(igrad.abs().max() for igrad in individual_gradients))
            xmin, xmax = -pad_factor * abs_max, pad_factor * abs_max

        elif self._adapt_policy == "min_max":
            min_val = float(min(igrad.min() for igrad in individual_gradients))
            max_val = float(max(igrad.max() for igrad in individual_gradients))
            span = max_val - min_val

            xmin = min_val - self._xpad * span
            xmax = max_val + self._xpad * span

        else:
            raise ValueError("Invalid adaptation policy")

        if xmax - xmin < self._min_xrange:
            warnings.warn(
                "Adaptive x limits are almost identical, using a small range instead."
            )
            center = (xmax + xmin) / 2
            xmin = center - self._min_xrange / 2
            xmax = center + self._min_xrange / 2

        self._xmin, self._xmax = xmin, xmax

    def _update_y_limits(self, params):
        """Update the histogram's y limits."""
        if self._adapt_policy == "abs_max":
            pad_factor = 1 + self._ypad
            abs_max = float(max(p.data.abs().max() for p in params))

            ymin, ymax = -pad_factor * abs_max, pad_factor * abs_max

        elif self._adapt_policy == "min_max":
            min_val = float(min(p.data.min() for p in params))
            max_val = float(max(p.data.max() for p in params))
            span = max_val - min_val

            ymin = min_val - self._ypad * span
            ymax = max_val + self._ypad * span

        else:
            raise ValueError("Invalid adaptation policy")

        if ymax - ymin < self._min_yrange:
            warnings.warn(
                "Adaptive y limits are almost identical, using a small range instead."
            )
            center = (ymax + ymin) / 2
            ymin = center - self._min_yrange / 2
            ymax = center + self._min_yrange / 2

        self._ymin, self._ymax = ymin, ymax


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
def test_grad_hist1d(problem, independent_runs):
    """Compare BackPACK and ``torch.autograd`` implementation of GradHist1d.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (GradHist1d, AutogradGradHist1d), schedule)


@pytest.mark.parametrize("problem", CPU_PROBLEMS, ids=CPU_PROBLEMS_ID)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
def test_grad_hist2d_few_bins_cpu(problem, independent_runs):
    """Compare BackPACK and ``torch.autograd`` implementation of GradHist2d.

    Use a small number of bins. This is because the histogram implementations
    have different floating precision inaccuracies. This leads to slightly
    deviating bin counts, which is hard to check. On GPUs, this problem becomes
    more pronounced.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.

    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)

    q_kwargs = {"xbins": 4, "ybins": 5}

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (GradHist2d, AutogradGradHist2d), schedule, q_kwargs=q_kwargs)
