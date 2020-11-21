"""Init for the two main parts of the Cockpit."""


from cockpit._version import __version__
from cockpit.cockpit import Cockpit
from cockpit.cockpit_plotter import CockpitPlotter

__all__ = ["Cockpit", "CockpitPlotter", "__version__"]
