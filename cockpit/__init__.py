"""Init for Cockpit."""
from pkg_resources import DistributionNotFound, get_distribution

from cockpit.cockpit import Cockpit
from cockpit.plotter import CockpitPlotter

# Extract the version number, for accessing it via __version__
try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


__all__ = ["Cockpit", "CockpitPlotter", "__version__", "cockpit.quantities"]
