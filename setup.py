"""Setup file for the Cockpit."""

import re
from os import path

from setuptools import find_packages, setup


# UTILS
##############################################################################
def _extract_version():
    with open(VERSIONFILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("__version__"):
                verstr = re.findall('"([^"]*)"', line)
    return verstr[0]


# META
##############################################################################
AUTHORS = "F. Schneider, F. Dangel"
NAME = "cockpit"
PACKAGES = find_packages()

DESCRIPTION = r"""A Cockpit for Training Neural Networks."""
LONG_DESCR = r"""Training neural networks using gradient-based methods is rather
an art than a science due to the lack of theoretical guarantees. It requires
intuition/experience gained from hours of hyperparameter tuning. Commonly, a
static or dynamic schedule is defined for all hyperparameters during training,
looping over a grid. Based on the learning curves of the test/train objective
and classification accuracy it is then decided, which setting is most
promising. While this procedure can be highly automated, it suffers from a poor
gain of insight in the optimization procedure. Almost never, quantities are
used during training to adapt the optimization strategy. We use BackPACK to
monitor quantities during training with the goal of both informing the machine
learning engineer about the state of training, and to identify useful rules for
when and how to change the optimization strategy if training stagnates."""

VERSIONFILE = "cockpit/_version.py"
VERSION = _extract_version()
URL = "https://github.com/f-dangel/backboard"
LICENSE = "MIT"

# DEPENDENCIES
##############################################################################
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_PATH = path.join(path.abspath(__file__), REQUIREMENTS_FILE)

setup(
    author=AUTHORS,
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCR,
    long_description_content_type="text/markdown",
    # install_requires=requirements,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
    python_requires=">=3.5",
)
