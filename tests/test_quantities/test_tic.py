"""Tests for ``backboard.quantities.tic.py``."""

import pytest

from backboard.quantities.tic import TICDiag, TICTrace
from tests.test_quantities.test_runner import run_sgd_test_runner
from tests.utils import hotfix_deepobs_argparse, set_deepobs_seed

TESTPROBLEMS = [
    "mnist_logreg",
    "mnist_mlp",
    "fmnist_logreg",
    "fmnist_mlp",
    # "fmnist_2c2d",
    # "mnist_2c2d",
    "cifar10_3c3d",
]

TRACK_INTERVAL = 1


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_tic_diag_precision(
    testproblem, num_epochs=1, batch_size=3, epsilon=1e-8, lr=0.01, momentum=0.0
):
    """Compare values of TICDiag computed with different extensions.

    Note: This test does not check if the value itself makes sense/is correct.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [
        TICDiag(
            TRACK_INTERVAL,
            curvature="diag_h",
            epsilon=epsilon,
            use_double=False,
            verbose=True,
            check=True,
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


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_tic_trace_precision(
    testproblem, num_epochs=1, batch_size=3, epsilon=1e-8, lr=0.01, momentum=0.0
):
    """Compare values of TICTrace computed with different extensions.

    Note: This test does not check if the value itself makes sense/is correct.
    """
    set_deepobs_seed(0)
    from backboard.utils import fix_deepobs_data_dir

    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    quantities = [
        TICTrace(
            TRACK_INTERVAL,
            curvature="diag_h",
            epsilon=epsilon,
            use_double=False,
            verbose=True,
            check=True,
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
