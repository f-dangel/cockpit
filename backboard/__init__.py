"""Init for the two main parts of the Cockpit."""


from backboard._version import __version__
from backboard.cockpit import Cockpit
from backboard.cockpit_plotter import CockpitPlotter

__all__ = ["Cockpit", "CockpitPlotter", "__version__"]
