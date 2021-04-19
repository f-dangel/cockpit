"""Fix for the automod extension."""

import os

from cockpit.instruments import __all__ as all_instruments
from cockpit.quantities import __all__ as all_quantities


def _create_quantities():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api/automod")

    if not os.path.exists(path):
        os.makedirs(path)

    for q in all_quantities:
        file_name = "cockpit.quantities." + q + ".rst"

        with open(os.path.join(path, file_name), "w+") as file:
            file.write(q + "\n")
            file.write(len(q) * "=" + "\n\n")

            file.write(".. currentmodule:: cockpit.quantities \n\n")
            file.write(".. autoclass:: " + q + "\n")


def _create_instruments():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api/automod")

    if not os.path.exists(path):
        os.makedirs(path)

    for q in all_instruments:
        file_name = "cockpit.instruments." + q + ".rst"

        with open(os.path.join(path, file_name), "w+") as file:
            file.write(q + "\n")
            file.write(len(q) * "=" + "\n\n")

            file.write(".. currentmodule:: cockpit.instruments \n\n")
            file.write(".. autoclass:: " + q + "\n")
