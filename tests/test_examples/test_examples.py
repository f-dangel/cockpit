"""Test whether the example scripts run."""

import os
import pathlib
import runpy
import sys
from unittest.mock import patch

import matplotlib as mpl
import pytest

mpl.use("Agg")

SCRIPTS = sorted(pathlib.Path(__file__, "../../..", "examples").resolve().glob("*.py"))
SCRIPTS_STR, SCRIPTS_ID = [], []
for s in SCRIPTS:
    if not str(s).split("/")[-1].startswith("_"):
        SCRIPTS_STR.append(str(s))
        SCRIPTS_ID.append(str(s).split("/")[-1].split(".")[0])


@patch("time.sleep")
@pytest.mark.parametrize("script", SCRIPTS_STR, ids=SCRIPTS_ID)
def test_example_scripts(mock, script):
    """Run a single example script.

    Args:
        mock (str): Call that should be replaced by a mock object.
        script (str): Script that should be run.
    """
    sys.path.append(os.path.dirname(script))
    del sys.argv[1:]  # Clear CLI arguments from pytest
    runpy.run_path(str(script))
