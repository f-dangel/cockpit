"""Define cockpit API."""


from cockpit._version import __version__
from cockpit.cockpit import Cockpit
from cockpit.plotter import CockpitPlotter

__all__ = ["Cockpit", "CockpitPlotter", "__version__", "cockpit.quantities"]
