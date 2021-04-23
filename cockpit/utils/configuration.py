"""Configuration utilities for cockpit."""

from cockpit import quantities
from cockpit.utils import schedules


def configuration(label, track_schedule=None, verbose=False):
    """Use pre-defined collections of quantities that should be used for tracking.

    Currently supports three different configurations:

    - ``"economy"``: Combines the :class:`~cockpit.quantities.Alpha`,
      :class:`~cockpit.quantities.Distance`, :class:`~cockpit.quantities.GradHist1d`,
      :class:`~cockpit.quantities.GradNorm`, :class:`~cockpit.quantities.InnerTest`,
      :class:`~cockpit.quantities.Loss`, :class:`~cockpit.quantities.NormTest`,
      :class:`~cockpit.quantities.OrthoTest` and :class:`~cockpit.quantities.UpdateSize`
      quantities.
    - ``"business"``: Same as ``"economy"`` but additionally with
      :class:`~cockpit.quantities.TICDiag` and :class:`~cockpit.quantities.HessTrace`.
    - ``"full"``: Same as ``"business"`` but additionally with
      :class:`~cockpit.quantities.HessMaxEV` and
      :class:`~cockpit.quantities.GradHist2d`.

    Args:
        label (str): String specifying the configuration type. Possible configurations
            are (least to most expensive) ``'economy'``, ``'business'``, ``'full'``.
        track_schedule (callable, optional): Function that maps the ``global_step``
            to a boolean, which determines if the quantity should be computed.
            Defaults to ``None``.
        verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.

    Returns:
        list: Instantiated quantities for a cockpit configuration.
    """
    if track_schedule is None:
        track_schedule = schedules.linear(interval=1, offset=0)

    quants = []
    for q_cls in quantities_cls_for_configuration(label):
        quants.append(q_cls(track_schedule=track_schedule, verbose=verbose))

    return quants


def quantities_cls_for_configuration(label):
    """Return the quantity classes for a cockpit configuration.

    Currently supports three different configurations:

    - ``"economy"``: Combines the :class:`~cockpit.quantities.Alpha`,
      :class:`~cockpit.quantities.Distance`, :class:`~cockpit.quantities.GradHist1d`,
      :class:`~cockpit.quantities.GradNorm`, :class:`~cockpit.quantities.InnerTest`,
      :class:`~cockpit.quantities.Loss`, :class:`~cockpit.quantities.NormTest`,
      :class:`~cockpit.quantities.OrthoTest` and :class:`~cockpit.quantities.UpdateSize`
      quantities.
    - ``"business"``: Same as ``"economy"`` but additionally with
      :class:`~cockpit.quantities.TICDiag` and :class:`~cockpit.quantities.HessTrace`.
    - ``"full"``: Same as ``"business"`` but additionally with
      :class:`~cockpit.quantities.HessMaxEV` and
      :class:`~cockpit.quantities.GradHist2d`.

    Args:
        label (str): String specifying the configuration type. Possible configurations
            are (least to most expensive) ``'economy'``, ``'business'``, ``'full'``.

    Returns:
        [Quantity]: A list of quantity classes used in the
        specified configuration.
    """
    economy = [
        quantities.Alpha,
        quantities.Distance,
        quantities.GradHist1d,
        quantities.GradNorm,
        quantities.InnerTest,
        quantities.Loss,
        quantities.NormTest,
        quantities.OrthoTest,
        quantities.UpdateSize,
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
