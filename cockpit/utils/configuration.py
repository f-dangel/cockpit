"""Configuration utilities for cockpit."""

from cockpit import quantities
from cockpit.utils.schedules import linear


def configuration(label, track_schedule=None, verbose=False):
    """Collect all quantities that should be used for tracking.

    Args:
        label (str): String specifying the configuration type. Possible configurations
            are (least to most expensive) ``'economy'``, ``'business'``, ``'full'``.
        track_schedule (callable): TODO
        verbose (bool): TODO

    Returns:
        list: Instantiated quantities for a cockpit configuration.
    """
    if track_schedule is None:
        track_schedule = linear(interval=1, offset=0)

    quants = []
    for q_cls in quantities_cls_for_configuration(label):
        quants.append(q_cls(track_schedule=track_schedule, verbose=verbose))

    return quants


def quantities_cls_for_configuration(label):
    """Return the quantity classes for a cockpit configuration.

    Args:
        label (str): String specifying the configuration type. Possible configurations
            are (least to most expensive) ``'economy'``, ``'business'``, ``'full'``.

    Returns:
        Quantity: A list of quantity classes used in the specified configuration.
    """
    economy = [
        quantities.Alpha,
        quantities.GradHist1d,
        quantities.Distance,
        quantities.UpdateSize,
        quantities.GradNorm,
        quantities.InnerTest,
        quantities.NormTest,
        quantities.OrthoTest,
        quantities.Loss,
    ]
    business = economy + [
        quantities.TICDiag,
        quantities.HessTrace,
    ]
    full = business + [
        quantities.HessMaxEV,
        quantities.GradHist2d,
    ]

    configs = {
        "full": full,
        "business": business,
        "economy": economy,
    }

    return configs[label]
