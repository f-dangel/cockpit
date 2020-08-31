"""Execute debugging script for orthogonality test.

Note: Changing the path of this file can lead to its failure.
"""

import os

from tests.utils import REPO_ROOT_DIR, run_command


def test_orthogonality_test_math_rearrangements():
    """Run tracking of orthogonality test and perform sanity checks during computation.

    Note: Running directly with ``pytest`` messes up ``argparse`` from DeepOBS.
    """
    target = os.path.join(REPO_ROOT_DIR, "exp/debug/track_orthogonality_test.py")
    run_command(["python", target])