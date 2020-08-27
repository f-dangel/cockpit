"""Execute debugging script for mean GSNR.

Note: Changing the path of this file can lead to its failure.
"""

import os

from tests.utils import REPO_ROOT_DIR, run_command


def test_mean_gsnr_precision_and_nans():
    """Run tracking of mean GSNR and perform sanity checks during computation.

    Note: Running directly with ``pytest`` messes up ``argparse`` from DeepOBS.
    """
    target = os.path.join(REPO_ROOT_DIR, "exp/debug/track_mean_gsnr.py")
    run_command(["python", target])
