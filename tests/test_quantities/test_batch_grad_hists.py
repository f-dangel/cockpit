"""Tests for individual gradient histograms."""

import pytest

from backboard.quantities import BatchGradHistogram1d, BatchGradHistogram2d
from tests.test_quantities.test_runner import run_sgd_test_runner
from tests.utils import hotfix_deepobs_argparse, set_deepobs_seed

TESTPROBLEMS = [
    "mnist_logreg",
    # "fmnist_2c2d",
    "mnist_mlp",
    "fmnist_logreg",
    "fmnist_mlp",
    # "mnist_2c2d",
    "cifar10_3c3d",
]

TRACK_INTERVAL = 1


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration_batch_grad_histogram_1d(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0
):
    """Integration test for individual gradient histograms.

    Note: This test only verifies that the computation passes.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [
        BatchGradHistogram1d(TRACK_INTERVAL, verbose=True, check=True),
    ]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration_batch_grad_histogram_2d(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0
):
    """Integration test for individual gradient histograms.

    Note: This test only verifies that the computation passes.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [
        BatchGradHistogram2d(TRACK_INTERVAL, verbose=True, check=True),
    ]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )
