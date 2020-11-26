"""Utility functions to configure cockpit."""

# TODO Remove DeepOBS dependency
from deepobs.config import set_data_dir


def fix_deepobs_data_dir():
    """Fix DeepOBS' data directory to one path avoid multiple dataset copies."""
    DIR = "~/tmp/data_deepobs"
    set_data_dir(os.path.expanduser(DIR))
