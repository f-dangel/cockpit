"""Tests for ``backboard.quantities.mean_gsnr.py``."""

import pytest

from backboard.quantities.mean_gsnr import MeanGSNR
from deepobs.config import set_data_dir
from tests.test_quantities.test_runner import run_sgd_test_runner
from tests.utils import hotfix_deepobs_argparse, set_deepobs_seed

TESTPROBLEMS = [
    "mnist_logreg",
    "mnist_mlp",
    "fmnist_logreg",
    "fmnist_mlp",
    # NOTE numerical stability issues, see comments above
    # "fmnist_2c2d",
    # "mnist_2c2d",
    # "cifar10_3c3d",
]

TRACK_INTERVAL = 1


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_mean_gsnr_precision_and_nans(
    testproblem, num_epochs=1, batch_size=3, epsilon=1e-5, lr=0.01, momentum=0.0
):
    """Compare values of GNSR computed with two different extensions.

    Verifies that the results don't have NaN values.

    The GSNR is introduced in equation (25) in liu2020understanding and recursively
    defined via the prose between Equation (1) and (2).

    Note: This test does not check if the value itself makes sense/is correct.
    """
    set_deepobs_seed(0)
    set_data_dir("~/tmp/data_deepobs")
    hotfix_deepobs_argparse()

    quantities = [
        MeanGSNR(
            TRACK_INTERVAL, epsilon=epsilon, use_double=False, verbose=True, check=True
        )
    ]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )
