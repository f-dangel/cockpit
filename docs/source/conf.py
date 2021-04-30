"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

from pkg_resources import DistributionNotFound, get_distribution

from cockpit.instruments import __all__ as all_instruments
from cockpit.quantities import __all__ as all_quantities

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "Cockpit"
author = "F. Schneider, F. Dangel"
copyright = str(datetime.utcnow().year) + " " + author

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
try:
    # The full version, including alpha/beta/rc tags.
    release = get_distribution(project).version
    # The short X.Y version.
    version = ".".join(release.split(".")[:2])
except DistributionNotFound:
    version = ""
finally:
    del get_distribution, DistributionNotFound


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "m2r2",
    "notfound.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    # "sphinx_copybutton",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Show __init__ member
autoclass_content = "both"

# Napoleon settings
napoleon_google_docstring = True

# Settings for automodapi
automodapi_toctreedirnm = "api/automod"
automodapi_writereprocessed = False
automodsumm_inherited_members = False
numpydoc_show_class_members = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Intersphinx configuration ----------------------------------------------

# Whenever Sphinx encounters a cross-reference that has no matching target in the
# current documentation set, it looks for targets in 'intersphinx_mapping'. A reference
# like :py:class:`zipfile.ZipFile` can then link to the Python documentation for the
# ZipFile class.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "PyTorch": ("https://pytorch.org/docs/master/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
}

# -- Notfound configuration ----------------------------------------------
notfound_context = {
    "title": "Page Not Found",
    "body": """
<h1>Page Not Found</h1>
<p>Sorry, we couldn't find that page.</p>
<p>Try using the search box or go to the homepage.</p>
""",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
html_theme_options = {"style_nav_header_background": "#fcfcfc"}

html_logo = "_static/LogoSquare.png"

html_favicon = "_static/favicon.ico"


# Fix for the automod extension.
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


# Use custom stylefile
def setup(app):
    """Add stylefile to rtd theme."""
    app.add_css_file("stylefile.css")
    _create_quantities()
    _create_instruments()
