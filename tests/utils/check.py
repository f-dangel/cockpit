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
    elif isinstance(value1, int) and isinstance(value2, int):
        compare_fn = compare_ints
    elif isinstance(value1, numpy.ndarray) and isinstance(value2, numpy.ndarray):
        compare_fn = compare_arrays
    elif isinstance(value1, list) and isinstance(value2, list):
        compare_fn = compare_lists
    elif isinstance(value1, tuple) and isinstance(value2, tuple):
        compare_fn = compare_tuples
    else:
        raise NotImplementedError(
            "No comparison available for these data types: "
            + f"{type(value1)}, {type(value2)}."
        )

    return compare_fn


def compare_tuples(tuple1, tuple2, rtol=1e-5, atol=1e-7):
    """Compare two tuples."""
    assert len(tuple1) == len(tuple2), "Different number of entries"

    for value1, value2 in zip(tuple1, tuple2):
        compare_fn = get_compare_function(value1, value2)
        compare_fn(value1, value2, rtol=rtol, atol=atol)


def compare_arrays(array1, array2, rtol=1e-5, atol=1e-7):
    """Compare two ``numpy`` arrays."""
    assert numpy.allclose(array1, array2, rtol=rtol, atol=atol)


def compare_floats(float1, float2, rtol=1e-5, atol=1e-7):
    """Compare two floats."""
    assert numpy.isclose(float1, float2, atol=atol, rtol=rtol), f"{float1} ≠ {float2}"


def compare_ints(int1, int2, rtol=None, atol=None):
    """Compare two integers.

    ``rtol`` and ``atol`` are ignored in the comparison, but required to keep the
    interface identical among comparison functions.

    Args:
        int1 (int): First integer.
        int2 (int): Another integer.
        rtol (any): Ignored, see above.
        atol (any): Ignored, see above.
    """
    assert int1 == int2


def compare_lists(list1, list2, rtol=1e-5, atol=1e-7):
    """Compare two lists containing floats."""
    assert len(list1) == len(
        list2
    ), f"Lists don't match in size: {len(list1)} ≠ {len(list2)}"

    for val1, val2 in zip(list1, list2):
        compare_fn = get_compare_function(val1, val2)
        compare_fn(val1, val2, rtol=rtol, atol=atol)
