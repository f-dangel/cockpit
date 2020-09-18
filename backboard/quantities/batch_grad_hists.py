"""Histograms of individual gradient transformations."""

import numpy
import torch

from backboard.quantities.quantity import Quantity
from backboard.quantities.utils_quantities import abs_max
from backpack import extensions


class BatchGradHistogram1d(Quantity):
    """One-dimensional histogram of individual gradient elements."""

    def __init__(
        self,
        track_interval,
        xmin=-2,
        xmax=2,
        bins=100,
        adapt_limits=True,
        adapt_limits_interval=None,
        verbose=False,
        check=False,
    ):
        """Initialize the 1D Histogram of individual gradient elements.

        Args:
            track_interval (int): Tracking rate.
            xmin (float): Lower clipping bound for individual gradients in histogram.
            xmax (float): Upper clipping bound for individual gradients in histogram.
            bins (int): Number of bins
            adapt_limits (bool): Adapt bin ranges.
            adapt_limits_interval (int): Adaptation frequency. Only used if
                ``update_limits`` is not ``None``. If ``None``, only adapt once.
                If ``-1``, use same value as ``track_interval``.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
            check (bool): If True, this quantity will be computed via two different
                ways and compared. Defaults to ``False``.

        """
        super().__init__(track_interval, verbose=verbose)
        self._xmin = xmin
        self._xmax = xmax
        self._bins = bins
        self._adapt_limits = adapt_limits

        if adapt_limits_interval == -1:
            self._adapt_limits_interval = track_interval
        else:
            self._adapt_limits_interval = adapt_limits_interval

        self._check = check

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if global_step % self._track_interval == 0:
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={"hist_1d": self._compute_histogram}
                )
            )

        if self._should_update_limits(global_step):
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
        if global_step % self._track_interval == 0:
            edges = self._get_current_bin_edges()
            hist = sum(p.grad_batch_transforms["hist_1d"] for p in params)

            if self._check:
                batch_size = self._fetch_batch_size_hotfix(batch_loss)
                num_params = sum(p.numel() for p in params)
                num_counts = hist.sum()
                assert batch_size * num_params == num_counts

            self.output[global_step]["hist_1d"] = hist.cpu().numpy().tolist()
            self.output[global_step]["edges"] = edges.cpu().numpy().tolist()

            if self._verbose:
                print(f"Histogram bin edges 0,...,10: {edges[:10]}")
                print(f"Histogram counts 0,...,10: {hist[:10]}")

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
        batch_size = batch_grad.shape[0]

        # clip to interval, elements outside [xmin, xmax] would be ignored
        batch_grad_clamped = torch.clamp(
            batch_size * batch_grad.data, self._xmin, self._xmax
        )

        return torch.histc(
            batch_grad_clamped, bins=self._bins, min=self._xmin, max=self._xmax
        )

    def _update_limits(self, global_step, params, batch_loss):
        """Update limits for next histogram computation."""
        if self._adapt_limits and self._should_update_limits(global_step):
            padding_factor = 1.2
            abs_max = padding_factor * max(
                p.grad_batch_transforms["grad_batch_abs_max"] for p in params
            )

            if self._verbose:
                print("Updating limits:")
                print(f"Old: x_min={self._xmin:.5f}, x_max={self._xmax:.5f}")

            self._xmin, self._xmax = -abs_max, abs_max

            if self._verbose:
                print(f"New: x_min={self._xmin:.5f}, x_max={self._xmax:.5f}")

    def _should_update_limits(self, global_step):
        """Return if current iteration should update the bin limits"""
        if self._adapt_limits_interval is None:
            is_first_update = global_step == 0
            return is_first_update
        else:
            return global_step % self._adapt_limits_interval == 0

    def _get_current_bin_edges(self):
        """Return current edge values of bins."""
        return torch.linspace(self._xmin, self._xmax, steps=self._bins + 1)


class BatchGradHistogram2d(Quantity):
    """Two-dimensional histogram of individual gradient elements over parameters.

    Individual gradient values are binned among the x-axis, parameter values are
    binned among the y-axis.
    """

    def __init__(
        self,
        track_interval,
        xmin=-1,
        xmax=1,
        xbins=100,
        ymin=-2,
        ymax=2,
        ybins=50,
        save_memory=True,
        adapt_limits=True,
        adapt_limits_interval=-1,
        verbose=False,
        check=False,
    ):
        """Initialize the 2D Histogram of individual gradient elements over parameters.

        Args:
            track_interval (int): Tracking rate.
            xmin (float): Lower clipping bound for individual gradients in histogram.
            xmax (float): Upper clipping bound for individual gradients in histogram.
            xbins (int): Number of bins in x-direction
            ymin (float): Lower clipping bound for parameters in histogram.
            ymax (float): Upper clipping bound for parameters in histogram.
            ybins (int): Number of bins in y-direction
            save_memory (bool): Sacrifice binning runtime for less memory.
            adapt_limits (bool): Adapt bin ranges.
            adapt_limits_interval (int): Adaptation frequency. Only used if
                ``update_limits`` is not ``None``. If ``None``, only adapt once.
                If ``-1``, use same value as ``track_interval``.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
            check (bool): If True, this quantity will be computed via two different
                ways and compared. Defaults to ``False``.

        """
        super().__init__(track_interval, verbose=verbose)
        self._xmin = xmin
        self._xmax = xmax
        self._xbins = xbins
        self._ymin = ymin
        self._ymax = ymax
        self._ybins = ybins
        self._save_memory = save_memory
        self._adapt_limits = adapt_limits

        if adapt_limits_interval == -1:
            self._adapt_limits_interval = track_interval
        else:
            self._adapt_limits_interval = adapt_limits_interval

        self._check = check

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if global_step % self._track_interval == 0:
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={"hist_2d": self._compute_histogram}
                )
            )

        if self._should_update_limits(global_step):
            ext.append(
                extensions.BatchGradTransforms(
                    transforms={
                        "grad_batch_abs_max": transform_grad_batch_abs_max,
                        "param_abs_max": transform_param_abs_max,
                    }
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
        if global_step % self._track_interval == 0:
            x_edges, y_edges = self._get_current_bin_edges()
            hist = sum(p.grad_batch_transforms["hist_2d"] for p in params)

            if self._check:
                batch_size = self._fetch_batch_size_hotfix(batch_loss)
                num_params = sum(p.numel() for p in params)
                num_counts = hist.sum()
                assert batch_size * num_params == num_counts

            self.output[global_step]["hist_2d"] = hist.cpu().numpy().tolist()
            self.output[global_step]["x_edges"] = x_edges.cpu().numpy().tolist()
            self.output[global_step]["y_edges"] = y_edges.cpu().numpy().tolist()

            if self._verbose:
                print(f"Histogram bin x_edges 0,...,5: {x_edges[:5]}")
                print(f"Histogram bin y_edges 0,...,5: {y_edges[:5]}")
                print(f"Histogram counts 0,...,5: {hist[:5,:5]}")

        self._update_limits(global_step, params, batch_loss)

    def _compute_histogram(self, batch_grad):
        """Transform individual gradients and parameters into a 2d histogram.

        Note:
            Currently, we have to compute multi-dimensional histograms with numpy.
            There is some activity to integrate such functionality into PyTorch here:
            https://github.com/pytorch/pytorch/issues/29209

        Todo:
            Wait for PyTorch functionality and replace numpy
        """
        batch_size = batch_grad.shape[0]

        # clip to interval, elements outside [xmin, xmax] would be ignored
        batch_grad_clamped = torch.clamp(
            batch_size * batch_grad.data, self._xmin, self._xmax
        )

        param = batch_grad._param_weakref().data
        param_clamped = torch.clamp(param, self._ymin, self._ymax)

        x_edges, y_edges = self._get_current_bin_edges()
        x_edges = x_edges.cpu().numpy()
        y_edges = y_edges.cpu().numpy()

        if self._save_memory:
            batch_grad_clamped = batch_grad_clamped.flatten(start_dim=1).cpu().numpy()
            param_clamped = param_clamped.flatten().cpu().numpy()

            hist = numpy.zeros(shape=(len(x_edges) - 1, len(y_edges) - 1))

            for n in range(batch_grad_clamped.shape[0]):
                h, xedges, yedges = numpy.histogram2d(
                    batch_grad_clamped[n], param_clamped, bins=(x_edges, y_edges)
                )
                hist += h

        else:
            expand_arg = [batch_size] + len(param.shape) * [-1]
            param_clamped = param_clamped.unsqueeze(0).expand(*expand_arg).flatten()

            batch_grad_clamped = batch_grad_clamped.flatten().cpu().numpy()
            param_clamped = param_clamped.cpu().numpy()

            hist, xedges, yedges = numpy.histogram2d(
                batch_grad_clamped, param_clamped, bins=(x_edges, y_edges)
            )

        if self._check:
            assert numpy.allclose(x_edges, xedges)
            assert numpy.allclose(y_edges, yedges)

        return torch.from_numpy(hist)

    def _update_limits(self, global_step, params, batch_loss):
        """Update limits for next histogram computation."""
        if self._adapt_limits and self._should_update_limits(global_step):
            self._update_x_limits(params)
            self._update_y_limits(params)

    def _update_x_limits(self, params):
        """Update the histogram's x limits."""
        padding_factor = 1.2
        abs_max = padding_factor * max(
            p.grad_batch_transforms["grad_batch_abs_max"] for p in params
        )

        if self._verbose:
            print("Updating limits:")
            print(f"Old: x_min={self._xmin:.5f}, x_max={self._xmax:.5f}")

        self._xmin, self._xmax = -abs_max, abs_max

        if self._verbose:
            print(f"New: x_min={self._xmin:.5f}, x_max={self._xmax:.5f}")

    def _update_y_limits(self, params):
        """Update the histogram's y limits."""
        padding_factor = 1.2
        abs_max = padding_factor * max(
            p.grad_batch_transforms["param_abs_max"] for p in params
        )

        if self._verbose:
            print("Updating limits:")
            print(f"Old: y_min={self._ymin:.5f}, y_max={self._ymax:.5f}")

        self._ymin, self._ymax = -abs_max, abs_max

        if self._verbose:
            print(f"New: y_min={self._ymin:.5f}, y_max={self._ymax:.5f}")

    def _should_update_limits(self, global_step):
        """Return if current iteration should update the bin limits"""
        if self._adapt_limits_interval is None:
            is_first_update = global_step == 0
            return is_first_update
        else:
            return global_step % self._adapt_limits_interval == 0

    def _get_current_bin_edges(self):
        """Return current edge values of bins."""
        x_edges = torch.linspace(self._xmin, self._xmax, steps=self._xbins + 1)
        y_edges = torch.linspace(self._ymin, self._ymax, steps=self._ybins + 1)
        return x_edges, y_edges


def transform_grad_batch_abs_max(batch_grad):
    """Compute maximum value of absolute individual gradients.

    Transformation used by BackPACK's ``BatchGradTransforms``.
    """
    batch_size = batch_grad.shape[0]
    return batch_size * abs_max(batch_grad.data).item()


def transform_param_abs_max(batch_grad):
    """Compute maximum value of absolute individual gradients.

    Transformation used by BackPACK's ``BatchGradTransforms``.
    """
    return abs_max(batch_grad._param_weakref().data)
