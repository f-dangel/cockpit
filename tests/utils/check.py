"""Utility functions to compare results of two implementations."""

import numpy


def compare_outputs(output1, output2, rtol=1e-5, atol=1e-7):
    """Compare outputs of two quantities."""
    assert len(list(output1.keys())) == len(
        list(output2.keys())
    ), "Different number of entries"

    for key in output1.keys():
        if isinstance(output1[key], dict):
            compare_outputs(output1[key], output2[key])
        else:
            val1, val2 = output1[key], output2[key]

            compare_fn = get_compare_function(val1, val2)

            compare_fn(val1, val2, atol=atol, rtol=rtol)


def get_compare_function(value1, value2):
    """Return the function used to compare ``value1`` with ``value2``."""
    if isinstance(value1, float) and isinstance(value2, float):
        compare_fn = compare_floats
    elif isinstance(value1, list) and isinstance(value2, list):
        compare_fn = compare_lists
    else:
        raise NotImplementedError(
            "No comparison available for these data types: "
            + f"{type(value1)}, {type(value2)}."
        )

    return compare_fn


def compare_floats(float1, float2, rtol=1e-5, atol=1e-7):
    """Compare two floats."""
    assert numpy.isclose(float1, float2, atol=atol, rtol=rtol), f"{float1} ≠ {float2}"


def compare_lists(list1, list2, rtol=1e-5, atol=1e-7):
    """Compare two lists containing floats."""
    assert len(list1) == len(
        list2
    ), f"Lists don't match in size: {len(list1)} ≠ {len(list2)}"

    for val1, val2 in zip(list1, list2):
        compare_fn = get_compare_function(val1, val2)
        compare_fn(val1, val2, rtol=rtol, atol=atol)
