"""Tests for ``backboard.quantities.hessian_eigenvalues``."""

import pytest

from backboard.quantities.max_ev import MaxEV
from tests.test_quantities.test_runner import run_sgd_test_runner
from tests.utils import hotfix_deepobs_argparse, set_deepobs_seed

TESTPROBLEMS = [
    "mnist_logreg",
    # "fmnist_2c2d",
    # "mnist_mlp",
    # "fmnist_logreg",
    # "fmnist_mlp",
    # "mnist_2c2d",
    # "cifar10_3c3d",
]

TRACK_INTERVAL = 1


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration_maximum_hessian_eigenvalue(
    testproblem, num_epochs=1, batch_size=2, lr=0.01, momentum=0.0
):
    """Integration test for maximum Hessian eigenvalue quantity.

    Computes maximum Hessian eigenvalue during a short training.
    Note: This test only verifies that the computation passes.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [MaxEV(TRACK_INTERVAL, verbose=True)]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )
