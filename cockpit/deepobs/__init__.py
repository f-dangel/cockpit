"""Integration to use cockpit on DeepOBS problems."""

# TODO Move to a cockpit_experiments library. Remove in public release.

# TODO Remove DeepOBS dependency
import os

from deepobs.config import set_data_dir


def fix_deepobs_data_dir():
    """Fix DeepOBS' data directory to one path avoid multiple dataset copies."""
    DIR = "~/tmp/data_deepobs"
    set_data_dir(os.path.expanduser(DIR))
