"""Tests for ``backboard.quantities.utils_hists.py``."""

import numpy
import pytest
import torch

from backboard.quantities.utils_hists import histogramdd

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_histogramdd(device):
    """Compare ``torch`` and ``numpy`` histogram function (d=2)."""
    torch.manual_seed(0)

    # set up
    N = 1000
    bins = 20

    x_data = torch.rand(N, device=device)
    y_data = torch.rand(N, device=device)

    # all values inside
    epsilon = 1e-6
    x_edges = torch.linspace(
        x_data.min() - epsilon, x_data.max() + epsilon, steps=bins + 1, device=device
    )
    y_edges = torch.linspace(
        y_data.min() - epsilon, y_data.max() + epsilon, steps=bins + 1, device=device
    )

    # run
    sample = torch.stack((x_data, y_data))

    torch_hist, _ = histogramdd(sample, bins=(x_edges, y_edges))

    numpy_hist, _, _ = numpy.histogram2d(
        x_data.cpu().numpy(),
        y_data.cpu().numpy(),
        bins=(x_edges.cpu().numpy(), y_edges.cpu().numpy()),
    )

    # compare
    torch_hist = torch_hist.cpu().int()
    numpy_hist = torch.from_numpy(numpy_hist).int()

    assert torch.allclose(torch_hist, numpy_hist)
