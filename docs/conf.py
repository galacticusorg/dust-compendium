"""Sphinx configuration."""

from dustcompendium import __version__

project = "Dust Compendium"
author = "Andrew Benson"
copyright = "2026, Andrew Benson"
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "sphinx_book_theme"
html_title = f"Dust Compendium {release}"

myst_enable_extensions = ["deflist", "dollarmath"]
