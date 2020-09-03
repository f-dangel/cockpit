"""Tests for ``backboard.quantities.orthogonality_test.py``.

Note: The orthogonality test seems to always pass in the first iterations
on mnist_logreg and fmnist_2c2d. Also, it suggests extremely small batch size.
I first thought there must be an error, but after a double check I could not
find anything. It seems that in the beginning of optimization, the gradients
are well-behaved.

Note: Similar to the norm test, the two tested ways to compute the orthogonality
    test deviate quite a lot numerically (only first three digits match when running
    on a CPU. The precision varies with hardware).
"""

import pytest

from backboard.quantities.orthogonality_test import OrthogonalityTest
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
def test_orthogonality_test_math(
    testproblem, num_epochs=1, batch_size=3, lr=0.01, momentum=0.0
):
    """Compare values of orthogonality test computed with two different extensions.

    Verifies correctness of the math rearrangement of the orthogonality test's
    original formulation (see Equation (3.3) in bollapragada2017adaptive).

    Note: This test does not check if the value itself makes sense/is correct.
    """
    set_deepobs_seed(0)
    set_data_dir("~/tmp/data_deepobs")
    hotfix_deepobs_argparse()

    quantities = [
        OrthogonalityTest(TRACK_INTERVAL, use_double=False, verbose=True, check=True)
    ]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )
