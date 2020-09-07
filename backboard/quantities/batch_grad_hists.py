"""Histograms of individual gradient transformations."""

import torch

from backboard.quantities.quantity import Quantity
from backpack import extensions


class BatchGradHistogram(Quantity):
    """Histogram of individual gradient elements."""

    def __init__(self, track_interval, verbose=False, check=False):
        super().__init__(track_interval, verbose=verbose)
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
            minimum = self.get_bins_min(global_step)
            maximum = self.get_bins_max(global_step)
            num_bins = self.get_num_bins(global_step)

            def compute_histogram(batch_grad):
                """Compute bin counts of individual gradients"""
                batch_size = batch_grad.size(0)

                # clip to interval, elements outside [minimum, maximum] would be ignored
                batch_grad_clamped = torch.clamp(
                    batch_size * batch_grad, minimum, maximum
                )

                return torch.histc(
                    batch_grad_clamped, bins=num_bins, min=minimum, max=maximum
                )

            ext.append(
                extensions.BatchGradTransforms(transforms={"hist": compute_histogram})
            )

        return ext

    def get_bins_min(self, global_step):
        """Return start value of bins at an iteration."""
        return -1

    def get_bins_max(self, global_step):
        """Return end value of bins at an iteration."""
        return 1

    def get_num_bins(self, global_step):
        """Return number of bins at an iteration."""
        return 100

    def get_bin_edges(self, global_step):
        """Return edge values of bins."""
        minimum = self.get_bins_min(global_step)
        maximum = self.get_bins_max(global_step)
        num_bins = self.get_num_bins(global_step)

        return torch.linspace(minimum, maximum, steps=num_bins + 1)

    def compute(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if global_step % self._track_interval == 0:
            edges = self.get_bin_edges(global_step)
            hist = sum(p.grad_batch_transforms["hist"] for p in params)

            if self._check:
                batch_size = self._fetch_batch_size_hotfix(batch_loss)
                num_params = sum(p.numel() for p in params)
                num_counts = hist.sum()
                assert batch_size * num_params == num_counts

            self.output[global_step]["hist"] = hist.cpu().numpy()
            self.output[global_step]["edges"] = edges.cpu().numpy()

            if self._verbose:
                print(f"Histogram bin edges 0,...,10: {edges[:10]}")
                print(f"Histogram counts 0,...,10: {hist[:10]}")

        else:
            pass
