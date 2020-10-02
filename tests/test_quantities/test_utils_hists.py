"""Tests for ``backboard.quantities.utils_hists.py``."""

import numpy
import pytest
import torch

from backboard.quantities.utils_hists import histogram2d, histogramdd

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


@pytest.mark.parametrize("device", DEVICES)
def test_histogram2d(device):
    """Compare ``torch`` and ``numpy`` 2d histogram function.

    Note:
        The conventions when a point is member of a bin differs bet-
        ween these functions. For large values of ``N`` one may find
        slighly different histograms.

    """
    torch.manual_seed(0)

    # set up
    N = 10000
    bins = (10, 5)

    x_data = torch.rand(N, device=device)
    y_data = torch.rand(N, device=device)

    # all values inside
    epsilon = 1e-6
    _range = (
        (x_data.min().item() - epsilon, x_data.max().item() + epsilon),
        (y_data.min().item() - epsilon, y_data.max().item() + epsilon),
    )

    # run
    sample = torch.stack((x_data, y_data))

    torch_hist, _ = histogram2d(sample, bins=bins, range=_range)

    numpy_hist, _, _ = numpy.histogram2d(
        x_data.cpu().numpy(), y_data.cpu().numpy(), bins=bins, range=_range
    )

    # compare
    torch_hist = torch_hist.cpu().int()
    numpy_hist = torch.from_numpy(numpy_hist).int()

    wrong = 0
    for idx, (h1, h2) in enumerate(zip(torch_hist.flatten(), numpy_hist.flatten())):
        if not torch.allclose(h1, h2):
            print(f"{idx}: {h1} ≠ {h2}")
            wrong += 1

    print(f"Mismatches: {wrong}")

    print(torch_hist.flatten()[:20])
    print(numpy_hist.flatten()[:20])

    assert torch.allclose(torch_hist, numpy_hist)


def test_simplistic_histogram2d():
    """Sanity check for two-dimensional histogram.

    Note:
        Histogram orientation is as follows:

        . → Y
        ↓
        X
    """
    xmin, xmax = 0, 3
    ymin, ymax = 0, 2

    xbins = 3
    ybins = 2

    # check versus numpy
    xedges = numpy.array([0.0, 1.0, 2.0, 3.0])
    yedges = numpy.array([0.0, 1.0, 2.0])

    xdata = numpy.array([0.5, 0.3, 1.5, 2.5])
    ydata = numpy.array([1.5, 1.3, 0.5, 0.5])

    hist = numpy.array(
        [
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    np_hist, np_xedges, np_yedges = numpy.histogram2d(
        xdata,
        ydata,
        bins=(xbins, ybins),
        range=((xmin, xmax), (ymin, ymax)),
    )

    assert numpy.allclose(np_xedges, xedges)
    assert numpy.allclose(np_yedges, yedges)
    assert numpy.allclose(np_hist, hist)

    # check versus torch
    xedges = torch.from_numpy(xedges).float()
    yedges = torch.from_numpy(yedges).float()

    xdata = torch.from_numpy(xdata)
    ydata = torch.from_numpy(ydata)

    hist = torch.from_numpy(hist).long()
    sample = torch.stack((xdata, ydata))

    torch_hist, (torch_xedges, torch_yedges) = histogram2d(
        sample,
        bins=(xbins, ybins),
        range=((xmin, xmax), (ymin, ymax)),
    )

    assert torch.allclose(torch_xedges, xedges)
    assert torch.allclose(torch_yedges, yedges)
    assert torch.allclose(torch_hist, hist)
