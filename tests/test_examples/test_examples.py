"""Test whether the example scripts run."""

import pathlib
import runpy
import sys
from unittest.mock import patch

import matplotlib as mpl
import pytest

mpl.use("Agg")

scripts = sorted(pathlib.Path(__file__, "../../..", "examples").resolve().glob("*.py"))
scripts_str, scripts_id = [], []
for s in scripts:
    scripts_str.append(str(s))
    scripts_id.append(str(s).split("/")[-1].split(".")[0])


@patch("time.sleep")
@pytest.mark.parametrize("scripts", scripts_str, ids=scripts_id)
def test_example_scripts(mock, scripts):
    """Run a single example script.

    Args:
        mock (str): Call that should be replaced by a mock object.
        scripts (str): Script that should be run.
    """
    del sys.argv[1:]  # Clear CLI arguments from pytest
    runpy.run_path(str(scripts))
