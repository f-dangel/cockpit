"""Utility Functions for the Quantities and the Tracking in General."""


def _update_dicts(master_dict, update_dict):
    """Merge dicts of dicts by updating dict a with dict b.

    Args:
        master_dict (dict): [description]
        update_dict (dict): [description]
    """
    for key, value in update_dict.items():
        for subkey, subvalue in value.items():
            master_dict[key][subkey] = subvalue
