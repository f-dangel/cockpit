"""Test the trim_dict function."""

import pytest

from backboard.tracking.utils_tracking import _trim_dict

dicts = []
dicts.append({"a": 4 * [0], "b": 3 * [0]})
dicts.append({"a": 3 * [0], "b": 2 * [0]})
dicts.append({"a": 4 * [0], "b": 3 * [0], "c": 4 * [1]})


@pytest.mark.parametrize("dict", dicts)
def test_trim_dict(dict):
    """Check whether valid dicts are trimmed to equal lenghts.

    Args:
        dict ([type]): A dictionary with lists as values. The lists must have
        either size `n` or `n+1`.
    """
    trimmed_dict = _trim_dict(dict)

    # Create dict where the values are the lenghts of the lists
    len_dicty = {key: len(value) for key, value in trimmed_dict.items()}

    # get a sorted list of the unique lenghts
    lengths = list(set(len_dicty.values()))
    lengths.sort()

    assert len(lengths) == 1


def test_trim_last_value():
    """Check whether the last value of a trimmed list will be removed."""
    dicty = {"a": 4 * [0], "b": 3 * [0], "c": [1, 2, 3, 4]}

    trimmed_dict = _trim_dict(dicty)

    assert trimmed_dict["c"] == dicty["c"][:-1]
