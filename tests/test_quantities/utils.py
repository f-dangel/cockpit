"""Utilities specifically required for testing quantities."""

import torch
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list

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
    """Instantiate problem, run ``SimpleTestHarness`` using ``quantities``.

    Return list of quantity outputs.
    """
    with instantiate(problem):
        testing_harness = SimpleTestHarness(problem)
        cockpit_kwargs = {"quantities": quantities}
        testing_harness.test(cockpit_kwargs)

    return [q.get_output() for q in quantities]


def autograd_hessian_columns(loss, params, concat=False):
    """Return an iterator of the Hessian columns computed via ``torch.autograd``.

    Args:
        concat (bool): If ``True``, flatten and concatenate the columns over all
            parameters.
    """
    D = sum(p.numel() for p in params)
    device = loss.device

    for d in range(D):
        e_d = torch.zeros(D, device=device)
        e_d[d] = 1.0
        e_d_list = vector_to_parameter_list(e_d, params)

        hessian_e_d = hessian_vector_product(loss, params, e_d_list)

        if concat:
            hessian_e_d = torch.cat([tensor.flatten() for tensor in hessian_e_d])

        yield hessian_e_d


def autograd_diag_hessian(loss, params, concat=False):
    """Compute the Hessian diagonal via ``torch.autograd``.

    Args:
        concat (bool): If ``True``, flatten and concatenate the columns over all
            parameters.
    """
    D = sum(p.numel() for p in params)
    device = loss.device

    hessian_diag = torch.zeros(D, device=device)

    for d, column_d in enumerate(autograd_hessian_columns(loss, params, concat=True)):
        hessian_diag[d] = column_d[d]

    if concat is False:
        hessian_diag = vector_to_parameter_list(hessian_diag, params)

    return hessian_diag
