"""Utilities specifically required for testing quantities."""

from tests.utils.check import compare_outputs
from tests.utils.harness import SimpleTestHarness
from tests.utils.problem import instantiate


def compare_quantities_single_run(problem, quantity_classes, schedule):
    """Run quantities in one cockpit. Compare written outputs."""
    q1_cls, q2_cls = quantity_classes
    q1 = q1_cls(track_schedule=schedule, verbose=True)
    q2 = q2_cls(track_schedule=schedule, verbose=True)

    outputs = run_harness_get_output(problem, [q1, q2])

    compare_outputs(outputs[0], outputs[1])


def compare_quantities_separate_runs(problem, quantity_classes, schedule):
    """Run quantities in separate cockpits. Compare written outputs."""
    assert (
        len(quantity_classes) == 2
    ), f"Can only compare 2 quantities, got {len(quantity_classes)}"

    outputs = []

    for q_cls in quantity_classes:
        q = q_cls(track_schedule=schedule, verbose=True)
        outputs.append(run_harness_get_output(problem, [q])[0])

    compare_outputs(outputs[0], outputs[1])


def run_harness_get_output(problem, quantities):
    with instantiate(problem):
        testing_harness = SimpleTestHarness(problem)
        cockpit_kwargs = {"quantities": quantities}
        testing_harness.test(cockpit_kwargs)

    return [q.get_output() for q in quantities]
