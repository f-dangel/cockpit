"""Tests for ``backboard.quantities.norm_test.py``.

Note:
   Computing the sample variance in two different ways can lead to surprisingly
   large differences of the results on CPUs when using large nets (for instance
   fmnist_2c2d, mnist_2c2d):
    - Only first three digits match when summing on CPU in single precision.
    - The first four to five digits match when summing on CPU in double precision
      (in the computation that we have access to, i.e. after the backward pass with
       BackPACK).
    - Both computations are stable (matching first six/seven digits) on GPU.
    - for 3c3d the precision is fine.
"""

import pytest

from backboard.quantities.norm_test import NormTest
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
def test_norm_test_math(testproblem, num_epochs=1, batch_size=3, lr=0.01, momentum=0.0):
    """Compare values of norm test computed with two different extensions.

    Verifies correctness of the math rearrangement of the norm test's original
    formulation (see Equation (3.9) in byrd2012sample).

    Note: This test does not check if the value itself makes sense/is correct.
    """
    set_deepobs_seed(0)
    set_data_dir("~/tmp/data_deepobs")
    hotfix_deepobs_argparse()

    quantities = [NormTest(TRACK_INTERVAL, use_double=False, verbose=True, check=True)]

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )
