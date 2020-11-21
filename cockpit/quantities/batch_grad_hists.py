"""Histograms of individual gradient transformations."""

import warnings

import numpy
import torch
from numpy import histogram2d as numpy_histogram2d

from backpack import extensions
from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_hists import (
    histogram2d,
    histogram2d_opt,
    histogramdd,
    transform_grad_batch_abs_max,
    transform_grad_batch_min_max,
    transform_param_abs_max,
    transform_param_min_max,
)


class BatchGradHistogram1d(SingleStepQuantity):
    """One-dimensional histogram of individual gradient elements."""

    def __init__(
        self,
        track_interval=1,
        track_offset=0,
        xmin=-2,
        xmax=2,
        bins=100,
        adapt_schedule=None,
        pad=0.2,
        verbose=False,
        check=False,
        track_schedule=None,
        remove_outliers=False,
    ):
        """Initialize the 1D Histogram of individual gradient elements.

        Args:
            track_interval (int): Tracking rate.
            xmin (float): Lower clipping bound for individual gradients in histogram.
            xmax (float): Upper clipping bound for individual gradients in histogram.
            bins (int): Number of bins
            adapt_schedule (callable): Function that maps ``global_step`` to a boolean
                that indicates if the limits should be updated. If ``None``, adapt
                only at step 0.
            pad (float): Relative padding added to the limits
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
            check (bool): If True, this quantity will be computed via two different
                ways and compared. Defaults to ``False``.
            remove_outliers (bool): Whether outliers should be removed. If ``False``,
                they show up in the edge bins.
        """
        super().__init__(
            track_interval=track_interval,
            track_offset=track_offset,
            verbose=verbose,
            track_schedule=track_schedule,
        )
        self._xmin = xmin
        self._xmax = xmax
        self._bins = bins
        self._pad = pad

        if adapt_schedule is None:

            def default_adapt_schedule(global_step):
                """Adapt at the very first step."""
                return global_step == 0

            self._adapt_schedule = default_adapt_schedule
        else:
            self._adapt_schedule = adapt_schedule

        self._check = check

        self._remove_outliers = remove_outliers

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.is_active(global_step):
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={"hist_1d": self._compute_histogram}
                )
            )

        if self._adapt_schedule(global_step):
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={"grad_batch_abs_max": transform_grad_batch_abs_max}
                )
            )

        return ext

    def compute(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            edges = self._get_current_bin_edges()
            hist = sum(p.grad_batch_transforms["hist_1d"] for p in params)

            if self._check:
                batch_size = get_batch_size(global_step)
                num_params = sum(p.numel() for p in params)
                num_counts = hist.sum()
                assert batch_size * num_params == num_counts

            self.output[global_step]["hist_1d"] = hist.cpu().numpy().tolist()
            self.output[global_step]["edges"] = edges.cpu().numpy().tolist()

            if self._verbose:
                print(
                    f"[Step {global_step}] BatchGradHistogram1d"
                    + f" edges 0,...,4: {edges[:5]}"
                )
                print(
                    f"[Step {global_step}] BatchGradHistogram1d"
                    + f" counts 0,...,4: {hist[:5]}"
                )

        self._update_limits(global_step, params, batch_loss)

    def _compute_histogram(self, batch_grad):
        """Transform individual gradients into histogram data.

        This function is be applied onto every parameter's individual gradient
        during backpropagation. If `p` denotes the parameter, then the return
        value is stored in the `"hist"` field of `p.grad_batch_transforms`.

        Args:
            batch_grad (torch.Tensor): Individual gradient of a parameter `p`. If
                `p` is of shape `(*)`, the individual gradients have shape `(N, *)`,
                where `N` denotes the batch size.
        """
        # NOTE ``batch_grad`` is 1/B ∇ℓᵢ so we need to compensate the 1/B. Instead of
        # multiplying ``batch_grad`` with the batch size, we instead scale the histogram
        # range to avoid an additional copy of batch_grad
        B = batch_grad.shape[0]

        return torch.histc(
            self.__preprocess(batch_grad),
            bins=self._bins,
            min=self._xmin / B,
            max=self._xmax / B,
        )

    def __preprocess(self, batch_grad):
        """Clip to histogram range if outliers should not be removed."""
        # NOTE ``batch_grad`` is 1/B ∇ℓᵢ so we need to compensate the 1/B. Instead of
        # multiplying ``batch_grad`` with the batch size, we instead scale the histogram
        # range to avoid an additional copy of batch_grad
        if self._remove_outliers:
            return batch_grad
        else:
            B = batch_grad.shape[0]

            # clip to interval, elements outside [xmin / B, xmax / B] would be ignored
            return torch.clamp(batch_grad.data, self._xmin / B, self._xmax / B)

    def _update_limits(self, global_step, params, batch_loss):
        """Update limits for next histogram computation."""
        if self._adapt_schedule(global_step):
            pad_factor = 1.0 + self._pad
            abs_max = pad_factor * max(
                p.grad_batch_transforms["grad_batch_abs_max"] for p in params
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
                    f"[Step {global_step}] BatchGradHistogram1d"
                    + f" new limits: ({self._xmin:.4f}, {self._xmax:.4f})",
                )

    def _get_current_bin_edges(self):
        """Return current edge values of bins."""
        return torch.linspace(self._xmin, self._xmax, steps=self._bins + 1)


class BatchGradHistogram2d(SingleStepQuantity):
    """Two-dimensional histogram of individual gradient elements over parameters.

    Individual gradient values are binned among the x-axis, parameter values are
    binned among the y-axis.
    """

    def __init__(
        self,
        track_interval=1,
        track_offset=0,
        xmin=-1,
        xmax=1,
        min_xrange=1e-6,
        xbins=40,
        ymin=-2,
        ymax=2,
        min_yrange=1e-6,
        ybins=50,
        save_memory=True,
        which="histogram2d",
        adapt_schedule=None,
        adapt_policy="abs_max",
        verbose=False,
        xpad=0.2,
        ypad=0.2,
        check=False,
        track_schedule=None,
        keep_individual=False,
    ):
        """Initialize the 2D Histogram of individual gradient elements over parameters.

        Args:
            track_interval (int): Tracking rate.
            xmin (float): Lower clipping bound for individual gradients in histogram.
            xmax (float): Upper clipping bound for individual gradients in histogram.
            min_xrange (float): Lower bound for limit difference along x axis.
            xbins (int): Number of bins in x-direction
            ymin (float): Lower clipping bound for parameters in histogram.
            ymax (float): Upper clipping bound for parameters in histogram.
            min_yrange (float): Lower bound for limit difference along y axis.
            ybins (int): Number of bins in y-direction
            save_memory (bool): Sacrifice binning runtime for less memory.
            which (str): Which histogram function should be used. Performance varies
                strongly among different methods, and also depend on the data being
                histogram-ed. Choices:
                - ``'numpy'``: Load to CPU and use ``numpy`` implementation.
                - ``'histogramdd'``: Use torch implementation that is currently under
                  review for being merged.
                - ``histogram2d``: Use custom torch implementation which uses ``put_``
                  instead of ``bincount``.
                - ``histogram2d_opt``: Use custom optimized torch implementation which
                  works without expanding the parameter values and saves some memory.
                  Will be unaffected by ``save_memory``.
            adapt_schedule (callable): Function that maps ``global_step`` to a boolean
                that indicates if the limits should be updated. If ``None``, adapt
                every time the histogram is recomputed.
            adapt_policy (str): Strategy to adapt the histogram limits.
                Options are:
                - "abs_max": Sets interval to range between negative and positive
                  maximum absolute value (+ padding).
                - "min_max": Sets interval range between minimum and maximum value
                  (+ padding).
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
            xpad (float): Relative padding added to the x limits.
            ypad (float): Relative padding added to the y limits.
            check (bool): If True, this quantity will be computed via two different
                ways and compared. Defaults to ``False``.
            keep_individual (bool): Whether to keep individual parameter histograms.
        """
        super().__init__(
            track_interval=track_interval,
            track_offset=track_offset,
            verbose=verbose,
            track_schedule=track_schedule,
        )
        self._xmin = xmin
        self._xmax = xmax
        self._min_xrange = min_xrange
        self._xbins = xbins
        self._ymin = ymin
        self._ymax = ymax
        self._min_yrange = min_yrange
        self._ybins = ybins
        self._save_memory = save_memory

        numpy.histogram2d.__name__ = "numpy_histogram2d"
        self.histogram_functions = {
            "numpy": numpy_histogram2d,
            "histogram2d": histogram2d,
            "histogram2d_opt": histogram2d_opt,
            "histogramdd": histogramdd,
        }
        assert which in self.histogram_functions.keys(), f"Invalid method {which}"

        self._which = which
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

        self._check = check

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.is_active(global_step):
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

    def compute(self, global_step, params, batch_loss):
        """Compute the two-dimensional histogram at the current iteration.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            self._compute_aggregated(global_step, params, batch_loss)

            if self._keep_individual:
                self._compute_individual(global_step, params, batch_loss)

        self._update_limits(global_step, params, batch_loss)

    def _compute_aggregated(self, global_step, params, batch_loss):
        """Aggregate histogram data over parameters and save to output."""
        x_edges, y_edges = self._get_current_bin_edges()
        hist = sum(p.grad_batch_transforms["hist_2d"] for p in params)

        if self._check:
            batch_size = get_batch_size(global_step)
            num_params = sum(p.numel() for p in params)
            num_counts = hist.sum()
            assert batch_size * num_params == num_counts

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

            if self._check:
                batch_size = get_batch_size(global_step)
                num_params = p.numel()
                num_counts = hist.sum()
                assert batch_size * num_params == num_counts

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
        if not self._which == "numpy":
            xedges, yedges = self._get_current_bin_edges()
            xbin_size, ybin_size = xedges[1] - xedges[0], yedges[1] - yedges[0]
            xepsilon, yepsilon = xbin_size / 2, ybin_size / 2

            xmin, xmax = xmin + xepsilon, xmax - xepsilon
            ymin, ymax = ymin + yepsilon, ymax - yepsilon

        batch_grad_clamped = torch.clamp(batch_size * batch_grad, xmin, xmax)
        param_clamped = torch.clamp(param, ymin, ymax)

        return batch_grad_clamped, param_clamped

    def __hist_save_mem(self, batch_grad_clamped, param_clamped):
        """Compute histogram and save memory.

        Note:
            Don't hand in sequences of arrays for ``bins`` as this way the
            histogram functions do not know that the bins are uniform. They
            will then call a sort algorithm, which is expensive.
        """
        batch_grad_clamped = batch_grad_clamped.flatten(start_dim=1)
        param_clamped = param_clamped.flatten()
        hist = torch.zeros(
            size=(self._xbins, self._ybins),
            device=param_clamped.device,
        )

        batch_size = batch_grad_clamped.shape[0]

        if self._which == "numpy":
            batch_grad_clamped = batch_grad_clamped.cpu().numpy()
            param_clamped = param_clamped.cpu().numpy()
            hist = hist.cpu().numpy()

        hist_bins = (self._xbins, self._ybins)
        hist_range = ((self._xmin, self._xmax), (self._ymin, self._ymax))
        hist_func = self.histogram_functions[self._which]

        if self._verbose:
            print(f"Using hist_func: {hist_func.__name__}")

        for n in range(batch_size):
            if self._which == "numpy":
                args = (batch_grad_clamped[n], param_clamped)
            else:
                args = (torch.stack((batch_grad_clamped[n], param_clamped)),)

            h = hist_func(*args, bins=hist_bins, range=hist_range)[0]
            hist += h

        if self._which == "numpy":
            hist = torch.from_numpy(hist)

        return hist

    def __hist_own_opt(self, batch_grad_clamped, param_clamped):
        """Custom optimized (individual gradient, parameter) 2d histogram."""
        hist_bins = (self._xbins, self._ybins)
        hist_range = ((self._xmin, self._xmax), (self._ymin, self._ymax))
        hist_func = self.histogram_functions[self._which]

        if self._verbose:
            print(f"Using hist_func: {hist_func.__name__}")

        hist = hist_func(
            batch_grad_clamped, param_clamped, bins=hist_bins, range=hist_range
        )[0]

        return hist

    def __hist_high_mem(self, batch_grad_clamped, param_clamped):
        """Compute histogram with memory-intensive strategy.

        Note:
            Don't hand in sequences of arrays for ``bins`` as this way the
            histogram functions do not know that the bins are uniform. They
            will then call a sort algorithm, which is expensive.
        """
        batch_size = batch_grad_clamped.shape[0]
        expand_arg = [batch_size] + len(param_clamped.shape) * [-1]
        param_clamped = param_clamped.unsqueeze(0).expand(*expand_arg).flatten()
        batch_grad_clamped = batch_grad_clamped.flatten()

        hist_bins = (self._xbins, self._ybins)
        hist_range = ((self._xmin, self._xmax), (self._ymin, self._ymax))
        hist_func = self.histogram_functions[self._which]

        if self._verbose:
            print(f"Using hist_func: {hist_func.__name__}")

        if self._which == "numpy":
            batch_grad_clamped = batch_grad_clamped.cpu().numpy()
            param_clamped = param_clamped.cpu().numpy()
            args = (batch_grad_clamped, param_clamped)
        else:
            args = (torch.stack((batch_grad_clamped, param_clamped)),)

        hist = hist_func(*args, bins=hist_bins, range=hist_range)[0]

        if self._which == "numpy":
            hist = torch.from_numpy(hist)

        return hist

    def _compute_histogram(self, batch_grad):
        """Transform individual gradients and parameters into a 2d histogram.

        Note:
            Currently, we have to compute multi-dimensional histograms with numpy.
            There is some activity to integrate such functionality into PyTorch here:
            https://github.com/pytorch/pytorch/issues/29209

        Todo:
            Wait for PyTorch functionality and replace numpy
        """
        batch_grad_clamped, param_clamped = self.__preprocess(
            batch_grad.data, batch_grad._param_weakref().data
        )

        if self._which == "histogram2d_opt":
            return self.__hist_own_opt(batch_grad_clamped, param_clamped)
        if self._save_memory:
            hist = self.__hist_save_mem(batch_grad_clamped, param_clamped)
        else:
            hist = self.__hist_high_mem(batch_grad_clamped, param_clamped)

        return hist

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
