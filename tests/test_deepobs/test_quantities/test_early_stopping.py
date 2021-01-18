"""Tests for `cockpit.quantities.early_stopping`."""

import pytest

from backpack import extensions
from cockpit.context import get_batch_size
from cockpit.quantities import EarlyStopping
from tests.test_quantities.utils import compare_quantities, get_output_sgd_test_runner

TESTPROBLEMS = [
    "quadratic_deep",
    "mnist_logreg",
    # "fmnist_2c2d",
    "mnist_mlp",
    "fmnist_logreg",
    "fmnist_mlp",
    # "mnist_2c2d",
    "cifar10_3c3d",
]

TRACK_INTERVAL = 2


class EarlyStoppingExpensive(EarlyStopping):
    """Evidence-based (EB) early-stopping criterion from individual gradients."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        if self.should_compute(global_step):
            ext = [extensions.BatchGrad()]
        else:
            ext = []

        return ext

    def _compute(self, global_step, params, batch_loss):
        """Compute the criterion.

        Evaluates the left hand side of Equ. 7 in

        - Mahsereci, M., Balles, L., Lassner, C., & Hennig, P.,
          Early stopping without a validation set (2017).

        If this value exceeds 0, training should be stopped.
        """
        B = get_batch_size(global_step)

        # compensate BackPACK's 1/B scaling
        batch_grad = B * self._fetch_batch_grad(params, aggregate=True)

        sgs_compensated = (batch_grad ** 2).sum(0)
        grad_squared = batch_grad.mean(0) ** 2

        diag_variance = (sgs_compensated - B * grad_squared) / (B - 1)

        snr = grad_squared / (diag_variance + self._epsilon)

        return 1 - B * snr.mean()


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Integration test for early stopping quantity.

    Note: This test only verifies that the computation passes.
    """
    quantities = [EarlyStopping(TRACK_INTERVAL, verbose=True)]

    return get_output_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
    )[0]


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_expensive_matches_separate_runs(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Compare with expensive early stopping criterion. Perform two runs."""
    quantity1 = EarlyStopping(TRACK_INTERVAL, verbose=True)
    quantity2 = EarlyStoppingExpensive(TRACK_INTERVAL, verbose=True)

    compare_quantities(
        [quantity1, quantity2],
        testproblem,
        separate_runs=True,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
    )


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_expensive_matches_joint_runs(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Compare with expensive early stopping criterion. Perform one run."""
    quantities = [
        EarlyStopping(TRACK_INTERVAL, verbose=True),
        EarlyStoppingExpensive(TRACK_INTERVAL, verbose=True),
    ]

    compare_quantities(
        quantities,
        testproblem,
        separate_runs=False,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
    )
