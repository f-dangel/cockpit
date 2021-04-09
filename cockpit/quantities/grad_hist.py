"""Histograms of individual gradient transformations."""

import warnings

import torch
from backpack import extensions

from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_hists import (
    histogram2d,
    transform_grad_batch_abs_max,
    transform_grad_batch_min_max,
    transform_param_abs_max,
    transform_param_min_max,
)


class GradHist1d(SingleStepQuantity):
    """One-dimensional histogram of individual gradient elements.

    Outlier elements are clipped to lie in the visible range.
    """

    def __init__(
        self,
        track_schedule,
        verbose=False,
        bins=100,
        range=(-2, 2),
    ):
        """Initialize the 1D Histogram of individual gradient elements.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
            bins (int): Number of bins
            range ((float, float), optional): Lower and upper limit of the bin range.
                Default: ``(-2, 2)``.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._range = range
        self._bins = bins

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.should_compute(global_step):
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={"hist_1d": self._compute_histogram}
                )
            )

        return ext

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
        hist = sum(p.grad_batch_transforms["hist_1d"][0] for p in params)
        edges = params[0].grad_batch_transforms["hist_1d"][1]

        return {"hist": hist, "edges": edges}

    def _compute_histogram(self, batch_grad):
        """Transform individual gradients into histogram data.

        This function is be applied onto every parameter's individual gradient
        during backpropagation. If `p` denotes the parameter, then the return
        value is stored in the `"hist"` field of `p.grad_batch_transforms`.

        Args:
            batch_grad (torch.Tensor): Individual gradient of a parameter `p`. If
                `p` is of shape `(*)`, the individual gradients have shape `(N, *)`,
                where `N` denotes the batch size.

        Returns:
            (torch.Tensor, torch.Tensor): First tensor represents histogram counts,
                second tensor are bin edges. Both are on the input's device.
        """
        # NOTE ``batch_grad`` is 1/B ∇ℓᵢ so we need to compensate the 1/B
        B = batch_grad.shape[0]
        individual_gradients = B * batch_grad

        start, end = self._range
        individual_gradients = torch.clamp(individual_gradients, start, end)

        hist = torch.histc(individual_gradients, bins=self._bins, min=start, max=end)
        edges = torch.linspace(start, end, self._bins + 1, device=batch_grad.device)

        return hist, edges


class GradHist2d(SingleStepQuantity):
    """Two-dimensional histogram of individual gradient elements over parameters.

    Individual gradient values are binned among the x-axis, parameter values are
    binned among the y-axis.
    """

    def __init__(
        self,
        track_schedule,
        verbose=False,
        xmin=-1,
        xmax=1,
        min_xrange=1e-6,
        xbins=40,
        ymin=-2,
        ymax=2,
        min_yrange=1e-6,
        ybins=50,
        adapt_schedule=None,
        adapt_policy="abs_max",
        xpad=0.2,
        ypad=0.2,
        keep_individual=False,
    ):
        """Initialize the 2D Histogram of individual gradient elements over parameters.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
            xmin (int, optional): Lower clipping bound for individual gradients
                in histogram. Defaults to -1.
            xmax (int, optional): Upper clipping bound for individual gradients
                in histogram. Defaults to 1.
            min_xrange (float, optional): Lower bound for limit difference along
                x axis. Defaults to 1e-6.
            xbins (int, optional): Number of bins in x-direction. Defaults to 40.
            ymin (int, optional): Lower clipping bound for parameters in histogram.
                Defaults to -2.
            ymax (int, optional): Upper clipping bound for parameters in histogram.
                Defaults to 2.
            min_yrange (float, optional): Lower bound for limit difference
                along y axis. Defaults to 1e-6.
            ybins (int, optional): Number of bins in y-direction. Defaults to 50.
            adapt_schedule (callable, optional): Function that maps ``global_step``
                to a boolean that indicates if the limits should be updated.
                If ``None``, adapt every time the histogram is recomputed.
                Defaults to None.
            adapt_policy (str, optional): Strategy to adapt the histogram limits.
                Options are:
                - "abs_max": Sets interval to range between negative and positive
                  maximum absolute value (+ padding).
                - "min_max": Sets interval range between minimum and maximum value
                  (+ padding).
                  Defaults to "abs_max".
            xpad (float, optional): Relative padding added to the x limits.
                Defaults to 0.2.
            ypad (float, optional): Relative padding added to the y limits.
                Defaults to 0.2.
            keep_individual (bool, optional):  Whether to keep individual
                parameter histograms. Defaults to False.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._xmin = xmin
        self._xmax = xmax
        self._min_xrange = min_xrange
        self._xbins = xbins
        self._ymin = ymin
        self._ymax = ymax
        self._min_yrange = min_yrange
        self._ybins = ybins

        self._xpad = xpad
        self._ypad = ypad
        self._keep_individual = keep_individual

        if adapt_schedule is None:
            self._adapt_schedule = self._track_schedule
        else:
            self._adapt_schedule = adapt_schedule

        assert adapt_policy in [
            "abs_max",
            "min_max",
        ], "Invalid adaptation policy"
        self._adapt_policy = adapt_policy

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Raises:
            ValueError: If unknown adaption policy.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.should_compute(global_step):
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={"hist_2d": self._compute_histogram}
                )
            )

        if self._adapt_schedule(global_step):
            if self._adapt_policy == "abs_max":
                ext.append(
                    extensions.BatchGradTransforms(
                        transforms={
                            "grad_batch_abs_max": transform_grad_batch_abs_max,
                            "param_abs_max": transform_param_abs_max,
                        }
                    )
                )
            elif self._adapt_policy == "min_max":
                ext.append(
                    extensions.BatchGradTransforms(
                        transforms={
                            "grad_batch_min_max": transform_grad_batch_min_max,
                            "param_min_max": transform_param_min_max,
                        }
                    )
                )
            else:
                raise ValueError("Invalid adaptation policy")

        return ext

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
        hist = sum(p.grad_batch_transforms["hist_2d"] for p in params)

        self.output[global_step]["hist_2d"] = hist.cpu().numpy().tolist()
        self.output[global_step]["x_edges"] = x_edges.cpu().numpy().tolist()
        self.output[global_step]["y_edges"] = y_edges.cpu().numpy().tolist()

        if self._verbose:
            print(
                f"[Step {global_step}] BatchGradHistogram2d"
                + f" x_edges 0,...,4: {x_edges[:5]}"
            )
            print(
                f"[Step {global_step}] BatchGradHistogram2d"
                + f" y_edges 0,...,4: {y_edges[:5]}"
            )
            print(
                f"[Step {global_step}] BatchGradHistogram2d"
                + f" counts [0,...,4][0,...,4]: {hist[:5,:5]}"
            )

    def _compute_individual(self, global_step, params, batch_loss):
        """Save histogram for each parameter to output."""
        for idx, p in enumerate(params):
            x_edges, y_edges = self._get_current_bin_edges()

            hist = p.grad_batch_transforms["hist_2d"]

            self.output[global_step][f"param_{idx}_hist_2d"] = (
                hist.cpu().numpy().tolist()
            )
            self.output[global_step][f"param_{idx}_x_edges"] = (
                x_edges.cpu().numpy().tolist()
            )
            self.output[global_step][f"param_{idx}_y_edges"] = (
                y_edges.cpu().numpy().tolist()
            )

            if self._verbose:
                print(
                    f"[Step {global_step}] BatchGradHistogram2d param_{idx}"
                    + f" x_edges 0,...,4: {x_edges[:5]}"
                )
                print(
                    f"[Step {global_step}] BatchGradHistogram2d param_{idx}"
                    + f" y_edges 0,...,4: {y_edges[:5]}"
                )
                print(
                    f"[Step {global_step}] BatchGradHistogram2d param_{idx}"
                    + f" counts [0,...,4][0,...,4]: {hist[:5,:5]}"
                )
        self.output[global_step]["param_groups"] = len(params)

    def __preprocess(self, batch_grad, param):
        """Scale and clamp the data used for histograms."""
        # clip to interval, elements outside [xmin, xmax] would be ignored
        batch_size = batch_grad.shape[0]

        xmin, xmax = self._xmin, self._xmax
        ymin, ymax = self._ymin, self._ymax

        # PyTorch implementation has different comparison conventions
        xedges, yedges = self._get_current_bin_edges()
        xbin_size, ybin_size = xedges[1] - xedges[0], yedges[1] - yedges[0]
        xepsilon, yepsilon = xbin_size / 2, ybin_size / 2

        xmin, xmax = xmin + xepsilon, xmax - xepsilon
        ymin, ymax = ymin + yepsilon, ymax - yepsilon

        batch_grad_clamped = torch.clamp(batch_size * batch_grad, xmin, xmax)
        param_clamped = torch.clamp(param, ymin, ymax)

        return batch_grad_clamped, param_clamped

    def __hist_high_mem(self, batch_grad_clamped, param_clamped):
        """Compute histogram with memory-intensive strategy.

        Note:
            Don't hand in sequences of arrays for ``bins`` as this way the
            histogram functions do not know that the bins are uniform. They
            will then call a sort algorithm, which is expensive.

        Args:
            batch_grad_clamped (Tensor): Clamped BatchGradients.
            param_clamped (Tensor): Clamped parameters.

        Returns:
            Tensor or NumpyArray: Histogram.
        """
        batch_size = batch_grad_clamped.shape[0]
        expand_arg = [batch_size] + len(param_clamped.shape) * [-1]
        param_clamped = param_clamped.unsqueeze(0).expand(*expand_arg).flatten()
        batch_grad_clamped = batch_grad_clamped.flatten()

        hist_bins = (self._xbins, self._ybins)
        hist_range = ((self._xmin, self._xmax), (self._ymin, self._ymax))
        hist_func = histogram2d

        if self._verbose:
            print(f"Using hist_func: {hist_func.__name__}")

        args = (torch.stack((batch_grad_clamped, param_clamped)),)

        hist = hist_func(*args, bins=hist_bins, range=hist_range)[0]

        return hist.float()

    def _compute_histogram(self, batch_grad):
        """Transform individual gradients and parameters into a 2d histogram.

        Note:
            Currently, we have to compute multi-dimensional histograms with numpy.
            There is some activity to integrate such functionality into PyTorch here:
            https://github.com/pytorch/pytorch/issues/29209

        Todo:
            Wait for PyTorch functionality and replace numpy

        Args:
            batch_grad (torch.Tensor): Individual gradient of a parameter `p`. If
                `p` is of shape `(*)`, the individual gradients have shape `(N, *)`,
                where `N` denotes the batch size.

        Returns:
            Callable, Tensor or NumpyArray: Histogram
        """
        batch_grad_clamped, param_clamped = self.__preprocess(
            batch_grad.data, batch_grad._param_weakref().data
        )

        return self.__hist_high_mem(batch_grad_clamped, param_clamped)

    def _update_limits(self, global_step, params, batch_loss):
        """Update limits for next histogram computation."""
        if self._adapt_schedule(global_step):
            self._update_x_limits(params)
            self._update_y_limits(params)

            if self._verbose:
                print(
                    f"[Step {global_step}] BatchGradHistogram2d"
                    + f" new x limits: ({self._xmin:.4f}, {self._xmax:.4f})",
                )
                print(
                    f"[Step {global_step}] BatchGradHistogram2d"
                    + f" new y limits: ({self._ymin:.4f}, {self._ymax:.4f})",
                )

    def _update_x_limits(self, params):
        """Update the histogram's x limits."""
        if self._adapt_policy == "abs_max":
            pad_factor = 1 + self._xpad
            abs_max = max(p.grad_batch_transforms["grad_batch_abs_max"] for p in params)
            xmin, xmax = -pad_factor * abs_max, pad_factor * abs_max

        elif self._adapt_policy == "min_max":
            min_val = min(
                p.grad_batch_transforms["grad_batch_min_max"][0] for p in params
            )
            max_val = max(
                p.grad_batch_transforms["grad_batch_min_max"][1] for p in params
            )
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
            abs_max = max(p.grad_batch_transforms["param_abs_max"] for p in params)

            ymin, ymax = -pad_factor * abs_max, pad_factor * abs_max

        elif self._adapt_policy == "min_max":
            min_val = min(p.grad_batch_transforms["param_min_max"][0] for p in params)
            max_val = max(p.grad_batch_transforms["param_min_max"][1] for p in params)
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

    def _get_current_bin_edges(self):
        """Return current edge values of bins."""
        x_edges = torch.linspace(self._xmin, self._xmax, steps=self._xbins + 1)
        y_edges = torch.linspace(self._ymin, self._ymax, steps=self._ybins + 1)
        return x_edges, y_edges
