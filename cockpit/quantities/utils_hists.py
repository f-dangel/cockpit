"""Utility functions for individual gradient histograms."""

import warnings

import torch

from cockpit.quantities.utils_quantities import abs_max

_range = range


def histogramdd(  # noqa: C901
    sample, bins=None, range=None, weights=None, remove_outliers=True
):
    """Create N-dimensional histograms.

    Args:
        sample (torch.Tensor): D by N Tensor, i-th row being the list of positions
            in the i-th coordinate, j-th column being the coordinates of j-th item
        bins (torch.Tensor, sequence or int, optional): (Will probably remove this)
            sequence of Tensors: Each Tensor defines the bins on each axis
            sequence of D ints: each int signifies the number of bins in each dimension
            int: signifies the number of ints in each dimension
        range (sequence, optional): Each item in the sequence is either a (min,max)
            Tuple, or None, in which case the edges are calculated as the minimum and
            maximum of input data
        weights (Tensor, optional): Weight for each value in the input tensor.
            Has to be ``None``. Defaults to ``None``.
        remove_outliers (bool, optional): Whether the overflow bins should be included
            in the final histogram or not

    Raises:
        ValueError: If the dimension of the bins is not equal to the dimension
            of the sample.
        ValueError: If the number of bins is not a positive integer.
        ValueError: If max of hist is not greater than min in range parameters.

    Returns:
        Histogram with axes
            * H              - Tensor - the histogram
            * axis           - list of Tensors - axis description

    Example:
        >>> r = torch.randn(3,100)
        >>> H,axes = histogramdd(r,bins = (4,3,7))
        >>> H.shape
        (4,3,7)

    Note:
        This code is a modified (with permission of the authors) version of
        - https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/
        Tools/Histograms/histogramdd_pytorch.py
    """
    edges = None
    device = None
    custom_edges = False
    D, N = sample.shape
    if device is None:
        device = sample.device
    if bins is None:
        if edges is None:
            bins = 10
            custom_edges = False
        else:
            try:
                bins = edges.size(1) - 1
            except AttributeError:
                bins = torch.empty(D)
                for i in _range(len(edges)):
                    bins[i] = edges[i].size(0) - 1
                bins = bins.to(device)
            custom_edges = True
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                "The dimension of bins must be equal to the dimension of the "
                " sample x."
            )
    except AttributeError:
        # bins is either an integer or a list
        if type(bins) == int:
            bins = torch.full([D], bins, dtype=torch.long, device=device)
        elif torch.is_tensor(bins[0]):
            custom_edges = True
            edges = bins
            bins = torch.empty(D, dtype=torch.long)
            for i in _range(len(edges)):
                bins[i] = edges[i].size(0) - 1
            bins = bins.to(device)
        else:
            bins = torch.as_tensor(bins).to(device)
    if bins.dim() == 2:
        custom_edges = True
        edges = bins
        bins = torch.full([D], bins.size(1) - 1, dtype=torch.long, device=device)
    if torch.any(bins <= 0):
        raise ValueError("The number of bins must be a positive integer.")
    if custom_edges:
        use_old_edges = False
        if not torch.is_tensor(edges):
            use_old_edges = True
            edges_old = edges
            m = max(i.size(0) for i in edges)
            tmp = torch.full([D, m], float("inf"), device=edges[0].device)
            for i in _range(D):
                s = edges[i].size(0)
                tmp[i, :] = edges[i][-1]
                tmp[i, :s] = edges[i][:]
            edges = tmp.to(device)
        warnings.warn(
            "Calling searchsorted is expensive and avoidable with uniform bins."
        )
        k = torch.searchsorted(edges, sample)
        k = torch.min(k, (bins + 1).reshape(-1, 1))
        if use_old_edges:
            edges = edges_old
        else:
            edges = torch.unbind(edges)
    else:
        if range is None:  # range is not defined
            range = torch.empty(2, D, device=device)
            if N == 0:  # Empty histogram
                range[0, :] = 0
                range[1, :] = 1
            else:
                range[0, :] = torch.min(sample, 1)[0]
                range[1, :] = torch.max(sample, 1)[0]
        elif not torch.is_tensor(range):  # range is a tuple
            r = torch.empty(2, D)
            for i in _range(D):
                if range[i] is not None:
                    r[:, i] = torch.as_tensor(range[i])
                else:
                    if N == 0:  # Edge case: empty histogram
                        r[0, i] = 0
                        r[1, i] = 1
                    else:
                        r[0, i] = torch.min(sample[:, i])
                        r[1, i] = torch.max(sample[:, i])
            range = r.to(device=device, dtype=sample.dtype)
        singular_range = torch.eq(
            range[0], range[1]
        )  # If the range consists of only one point, pad it up
        range[0, singular_range] -= 0.5
        range[1, singular_range] += 0.5
        if torch.any(range[0] > range[1]):
            raise ValueError("Max must be greater than min in range parameters.")
        edges = [
            torch.linspace(range[0, i], range[1, i], bins[i] + 1)
            for i in _range(len(bins))
        ]
        tranges = torch.empty_like(range)
        tranges[1, :] = bins / (range[1, :] - range[0, :])
        tranges[0, :] = 1 - range[0, :] * tranges[1, :]
        k = torch.addcmul(
            tranges[0, :].reshape(-1, 1), sample, tranges[1, :].reshape(-1, 1)
        ).long()  # Get the right index
        k = torch.max(
            k, torch.zeros([], device=device, dtype=torch.long)
        )  # Underflow bin
        k = torch.min(k, (bins + 1).reshape(-1, 1))

    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:], [0]) + 2, -1).long()
    multiindex = torch.flip(multiindex, [0])
    k *= multiindex.reshape(-1, 1)
    l = torch.sum(k, 0)  # noqa: E741

    assert weights is None, "weights has to be None"

    hist = torch.bincount(l, minlength=(multiindex[0] * (bins[0] + 2)), weights=weights)
    hist = hist.reshape(tuple(bins + 2))

    if remove_outliers:
        core = D * (slice(1, -1),)
        hist = hist[core]

    return hist, edges


def histogram2d(sample, bins, range, check_input=False):
    """Compute a two-dimensional histogram."""
    xbins, ybins = bins
    (xmin, xmax), (ymin, ymax) = range

    if check_input:
        assert sample.shape[0] == 2
        assert len(sample.shape) == 2

        assert (
            sample[0].min() > xmin
        ), f"sample[0] min too small: {sample[0].min()} > {xmin}"
        assert (
            sample[0].max() < xmax
        ), f"sample[0] max too big: {sample[0].max()} > {xmax}"

        assert (
            sample[1].min() > ymin
        ), f"sample[1] min too small: {sample[1].min()} > {ymin}"
        assert (
            sample[1].max() < ymax
        ), f"sample[1] max too big: {sample[1].max()} > {ymax}"

    xbin_width = (xmax - xmin) / xbins
    ybin_width = (ymax - ymin) / ybins

    idx = ybins * ((sample[0] - xmin) / xbin_width).long()
    idx += ((sample[1] - ymin) / ybin_width).long()

    one_shape = tuple(1 for _ in idx.shape)
    expand_shape = tuple(idx.shape)
    ones = torch.ones(one_shape, dtype=int, device=idx.device).expand(*expand_shape)

    hist = torch.zeros(xbins * ybins, dtype=int, device=sample.device)
    hist.put_(idx, ones, accumulate=True)

    axes = (
        torch.linspace(xmin, xmax, steps=xbins + 1, device=sample.device),
        torch.linspace(ymin, ymax, steps=ybins + 1, device=sample.device),
    )

    return hist.reshape(xbins, ybins), axes


def histogram2d_opt(batch_grad, param, bins, range, check_input=False):
    """Compute a two-dimensional histogram."""
    xbins, ybins = bins
    (xmin, xmax), (ymin, ymax) = range

    if check_input:
        assert (
            batch_grad.min() > xmin
        ), f"batch_grad min too small: {batch_grad.min()} > {xmin}"
        assert (
            batch_grad.max() < xmax
        ), f"batch_grad max too big: {batch_grad.max()} > {xmax}"

        assert param.min() > ymin, f"param min too small: {param.min()} > {ymin}"
        assert param.max() < ymax, f"param max too big: {param.max()} > {ymax}"

    xbin_width = (xmax - xmin) / xbins
    ybin_width = (ymax - ymin) / ybins

    idx = ybins * ((batch_grad - xmin) / xbin_width).long()
    idx += ((param - ymin) / ybin_width).long()

    one_shape = tuple(1 for _ in idx.shape)
    expand_shape = tuple(idx.shape)
    ones = torch.ones(one_shape, dtype=int, device=idx.device).expand(*expand_shape)

    hist = torch.zeros(xbins * ybins, dtype=int, device=idx.device)
    hist.put_(idx, ones, accumulate=True)

    axes = (
        torch.linspace(xmin, xmax, steps=xbins + 1, device=idx.device),
        torch.linspace(ymin, ymax, steps=ybins + 1, device=idx.device),
    )

    return hist.reshape(xbins, ybins), axes


def transform_grad_batch_abs_max(batch_grad):
    """Compute maximum value of absolute individual gradients.

    Transformation used by BackPACK's ``BatchGradTransforms``.

    Args:
        batch_grad (torch.Tensor): Individual gradient of a parameter `p`. If
                `p` is of shape `(*)`, the individual gradients have shape `(N, *)`,
                where `N` denotes the batch size.

    Returns:
        float: Maximum value of absolute individual gradients.
    """
    batch_size = batch_grad.shape[0]
    return batch_size * abs_max(batch_grad.data).item()


def transform_param_abs_max(batch_grad):
    """Compute maximum value of absolute individual gradients.

    Transformation used by BackPACK's ``BatchGradTransforms``.

    Args:
        batch_grad (torch.Tensor): Individual gradient of a parameter `p`. If
                `p` is of shape `(*)`, the individual gradients have shape `(N, *)`,
                where `N` denotes the batch size.

    Returns:
        float: Maximum value of absolute individual gradients.
    """
    return abs_max(batch_grad._param_weakref().data).item()


def transform_grad_batch_min_max(batch_grad):
    """Compute minimum and maximum value of absolute individual gradients.

    Transformation used by BackPACK's ``BatchGradTransforms``.

    Args:
        batch_grad (torch.Tensor): Individual gradient of a parameter `p`. If
                `p` is of shape `(*)`, the individual gradients have shape `(N, *)`,
                where `N` denotes the batch size.

    Returns:
        list: Minimum and maximum value of absolute individual gradients.
    """
    batch_size = batch_grad.shape[0]
    return [
        batch_size * batch_grad.data.min().item(),
        batch_size * batch_grad.data.max().item(),
    ]


def transform_param_min_max(batch_grad):
    """Compute minimum and maximum value of absolute individual gradients.

    Transformation used by BackPACK's ``BatchGradTransforms``.

    Args:
        batch_grad (torch.Tensor): Individual gradient of a parameter `p`. If
                `p` is of shape `(*)`, the individual gradients have shape `(N, *)`,
                where `N` denotes the batch size.

    Returns:
        list: Minimum and maximum value of absolute individual gradients.
    """
    return [
        batch_grad._param_weakref().data.min().item(),
        batch_grad._param_weakref().data.max().item(),
    ]
