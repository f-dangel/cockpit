"""Setup file for python_package_template using setup.cfg for configuration."""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup

# Use setup.cfg if possible
try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup()
