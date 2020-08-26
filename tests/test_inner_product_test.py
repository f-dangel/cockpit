"""Execute debugging script for inner product test.

Note: Changing the path of this file can lead to its failure.
"""

import os

from tests.utils import run_command

REPO_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_inner_product_test_math_rearrangements():
    """Run tracking of inner product test and perform sanity checks during computation.

    Note: Running directly with ``pytest`` messes up ``argparse`` from DeepOBS.
    """
    target = os.path.join(REPO_ROOT_DIR, "exp/debug/track_inner_product_test.py")
    run_command(["python", target])
