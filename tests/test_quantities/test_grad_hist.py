"""Compare ``GradHist{1,2}d`` quantities with ``torch.autograd``."""

import numpy
import pytest
import torch

from cockpit.context import get_individual_losses
from cockpit.quantities import GradHist1d, GradHist2d
from tests.test_quantities.settings import (
    CPU_PROBLEMS,
    CPU_PROBLEMS_ID,
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
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
        if self._adapt is None:
            should_adapt = False
        else:
            should_adapt = self._adapt.should_compute(global_step)

        return self.should_compute(global_step) or should_adapt

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the individual gradient histogram.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            dict: Entry ``'hist'`` holds the histogram, entry ``'edges'`` holds
                the bin limits.
        """
        individual_losses = get_individual_losses(global_step)
        individual_gradients = autograd_individual_gradients(
            individual_losses, params, concat=True
        )
        hist, edges = self._compute_histogram(individual_gradients)

        return {"hist": hist.float(), "edges": edges}

    def _compute_histogram(self, individual_gradients):
        """Compute bin counts of individual gradient elements.

        Args:
            individual_gradients (torch.Tensor): Tensor holding individual gradients.

        Returns:
            (torch.Tensor, torch.Tensor): First tensor represents histogram counts,
                second tensor are bin edges. Both are on the input's device.
        """
        data = torch.clamp(individual_gradients, *self._range).detach().cpu().numpy()

        hist, edges = numpy.histogram(data, bins=self._bins, range=self._range)

        # convert to torch and load to device
        device = individual_gradients.device

        hist = torch.from_numpy(hist).to(device)
        edges = torch.from_numpy(edges).to(device)

        return hist, edges


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
        should_adapt = any(
            a.should_compute(global_step) for a in self._adapt if a is not None
        )

        return self.should_compute(global_step) or should_adapt

    def _compute(self, global_step, params, batch_loss):
        """Aggregate histogram data over parameters and save to output."""
        individual_losses = get_individual_losses(global_step)
        individual_gradients = autograd_individual_gradients(individual_losses, params)
        layerwise = [
            self._compute_histogram(p, igrad)
            for p, igrad in zip(params, individual_gradients)
        ]

        hist = sum(out[0] for out in layerwise)
        edges = layerwise[0][1]

        result = {"hist": hist, "edges": edges}

        if self._keep_individual:
            result["param_groups"] = len(params)

            for idx, (hist, edges) in enumerate(layerwise):
                result[f"param_{idx}"] = {"hist": hist, "edges": edges}

        return result

    def _compute_histogram(self, param, individual_gradients):
        """Compute bin counts of individual gradient elements."""
        batch_size = individual_gradients.shape[0]

        data = [individual_gradients, param]

        for dim, data_dim in enumerate(data):
            lower, upper = self._range[dim]
            bins = self._bins[dim]

            # Histogram implementation does not include the limits, clip to bin center
            bin_size = (upper - lower) / bins
            data[dim] = torch.clamp(
                data_dim, min=lower + bin_size / 2, max=upper - bin_size / 2
            )

        hist = None
        edges = None
        device = individual_gradients.device

        for b in range(batch_size):

            hist_b, xedges, yedges = numpy.histogram2d(
                data[0][b].detach().flatten().cpu().numpy(),
                data[1].detach().flatten().cpu().numpy(),
                bins=self._bins,
                range=self._range,
            )

            hist_b = torch.from_numpy(hist_b).long().to(device)
            xedges = torch.from_numpy(xedges).float().to(device)
            yedges = torch.from_numpy(yedges).float().to(device)

            hist = hist_b if hist is None else hist + hist_b
            edges = (xedges, yedges) if edges is None else edges

        return hist, edges


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_grad_hist1d(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of GradHist1d.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (GradHist1d, AutogradGradHist1d), q_kwargs)


@pytest.mark.parametrize("problem", CPU_PROBLEMS, ids=CPU_PROBLEMS_ID)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_grad_hist2d_few_bins_cpu(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of GradHist2d.

    Use a small number of bins. This is because the histogram implementations
    have different floating precision inaccuracies. This leads to slightly
    deviating bin counts, which is hard to check. On GPUs, this problem becomes
    more pronounced.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    q_extra_kwargs = {"bins": (4, 5)}
    q_extra_kwargs = {"bins": (8, 5), "range": ((-0.01, 0.01), (-0.02, 0.02))}
    combined_q_kwargs = {**q_kwargs, **q_extra_kwargs}

    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (GradHist2d, AutogradGradHist2d), combined_q_kwargs)
