"""Setup file for the Cockpit."""

from os import path

from setuptools import find_packages, setup

# META
##############################################################################
AUTHORS = "F. Schneider, F. Dangel"
NAME = "backboard-for-pytorch"
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

VERSION = "0.0.1"
URL = "https://github.com/f-dangel/backboard"
LICENSE = "MIT"

# DEPENDENCIES
##############################################################################
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_PATH = path.join(path.abspath(__file__), REQUIREMENTS_FILE)

# with open(REQUIREMENTS_FILE) as f:
#    requirements = f.read().splitlines()

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
