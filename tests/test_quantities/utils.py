"""Utility function for tests of ``backboard.quantities``."""

import numpy

from tests.test_quantities.test_runner import run_sgd_test_runner
from tests.utils import hotfix_deepobs_argparse, set_deepobs_seed


def compare_outputs(output1, output2, rtol=5e-4, atol=1e-5):
    """Compare outputs of two quantities."""
    assert len(list(output1.keys())) == len(
        list(output2.keys())
    ), "Different number of entries"

    for key in output1.keys():
        if isinstance(output1[key], dict):
            compare_outputs(output1[key], output2[key])
        else:
            val1, val2 = output1[key], output2[key]
            if isinstance(val1, float) and isinstance(val2, float):
                assert numpy.isclose(val1, val2, atol=atol, rtol=rtol)
            else:
                raise NotImplementedError("No comparison available for this data type.")


def compare_quantities(
    quantities,
    testproblem,
    separate_runs=True,
    num_epochs=1,
    batch_size=4,
    lr=0.01,
    momentum=0.0,
    seed=0,
):
    """Compare output of two quantities. Allow running separately to avoid interaction.

    Note:
        This may lead to problems if the quantities themselves use random numbers.
    """
    assert len(quantities) == 2, "Can only compare two quantities"

    if separate_runs:
        output1, output2 = [
            get_output_sgd_test_runner(
                [q],
                testproblem,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                seed=seed,
            )[0]
            for q in quantities
        ]
    else:
        output1, output2 = [
            get_output_sgd_test_runner(
                quantities,
                testproblem,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                seed=seed,
            )[0]
            for q in quantities
        ]

    compare_outputs(output1, output2)


def get_output_sgd_test_runner(
    quantities, testproblem, num_epochs=1, batch_size=4, lr=0.01, momentum=0.0, seed=0
):
    from backboard.utils import fix_deepobs_data_dir

    set_deepobs_seed(seed)
    fix_deepobs_data_dir()
    hotfix_deepobs_argparse()

    run_sgd_test_runner(
        quantities,
        testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
    )

    return [q.output for q in quantities]
