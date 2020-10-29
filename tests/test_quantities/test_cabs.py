"""Tests for `backboard.quantities.early_stopping`."""

import pytest

from backboard.context import get_batch_size
from backboard.quantities import CABS
from backpack import extensions
from tests.test_quantities.utils import compare_quantities, get_output_sgd_test_runner

TESTPROBLEMS = [
    "quadratic_deep",
    "mnist_logreg",
    "fmnist_2c2d",
    "mnist_mlp",
    "fmnist_logreg",
    "fmnist_mlp",
    "mnist_2c2d",
    "cifar10_3c3d",
]

TRACK_INTERVAL = 2


class CABSExpensive(CABS):
    """CABS rule from individual gradients."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []
        if self.is_active(global_step):
            ext.append(extensions.BatchGrad())

        return ext

    def _compute(self, global_step, params, batch_loss):
        """Compute the CABS rule. Return suggested batch size.

        Evaluates Equ. 22 of

        - Balles, L., Romero, J., & Hennig, P.,
          Coupling adaptive batch sizes with learning rates (2017).
        """
        B = get_batch_size(global_step)

        grad_squared = self._fetch_grad(params, aggregate=True) ** 2
        # # compensate BackPACK's 1/B scaling
        batch_grad_compensated = B * self._fetch_batch_grad(params, aggregate=True)

        sgs = (batch_grad_compensated ** 2).sum()
        ssg = grad_squared.sum()

        return self._lr * (sgs - B * ssg) / (B * batch_loss.item())


@pytest.mark.parametrize("testproblem", TESTPROBLEMS, ids=TESTPROBLEMS)
def test_integration(
    testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    """Integration test for early stopping quantity.

    Note: This test only verifies that the computation passes.
    """
    quantities = [CABS(TRACK_INTERVAL, verbose=True)]

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
    quantity1 = CABS(TRACK_INTERVAL, verbose=True)
    quantity2 = CABSExpensive(TRACK_INTERVAL, verbose=True)

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
        CABS(TRACK_INTERVAL, verbose=True),
        CABSExpensive(TRACK_INTERVAL, verbose=True),
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
