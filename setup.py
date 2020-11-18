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

DESCRIPTION = r"""A Practical Debugging Tool for Training Deep Neural Networks."""
LONG_DESCR = r"""When engineers train deep learning models, they are very much
“flying blind”. Commonly used approaches for real-time training diagnostics,
such as monitoring the train/test loss, are limited. Assessing a network's
training process solely through these performance indicators is akin to
debugging software without access to internal states through a debugger. To
address this, we present Cockpit, a collection of instruments that enable a
closer look into the inner workings of a learning machine, and a more
informative and meaningful status report for practitioners. It facilitates the 
identification of learning phases and failure modes, like ill-chosen
hyperparameters. These instruments leverage novel higher-order information about
the gradient distribution and curvature, which has only recently become
efficiently accessible. We believe that such a debugging tool, which we
open-source for PyTorch, represents an important step to improve troubleshooting
the training process, reveal new insights, and help develop novel methods and
heuristics."""

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
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
    python_requires=">=3.5",
)
