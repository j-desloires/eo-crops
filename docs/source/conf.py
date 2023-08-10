# -*- coding: utf-8 -*-  # noqa: UP009
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import shutil
import sys

import sys

sys.path.insert(0, os.path.abspath("../../"))

print("Python Version:", sys.version)
print("sys.path:", sys.path)


import eocrops
from collections import defaultdict
from typing import Any, Dict, Optional

import sphinx.ext.autodoc

# -- Project information -----------------------------------------------------

# General information about the project.
project = "EOCrops"
copyright = "2023, INRAE"  # noqa: A001
author = "Johann Desloires"
doc_title = "EOCrops Documentation"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The release is read from __init__ file and version is shortened release string.
with open(os.path.join(os.path.dirname(__file__), "../../setup.py")) as setup_file:
    for line in setup_file:
        if "version=" in line:
            release = line.split("=")[1].strip('", \n').strip("'")
            version = release.rsplit(".", 1)[0]

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_nested_apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx_mdinclude",
    "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinxcontrib.jquery",
]

# Name of the package directory.
sphinx_nested_apidoc_package_dir = "eocrops"
# Name of the folder to put all the package documentation in. By default it is
# the name of the package itself.
sphinx_nested_apidoc_package_name = "eocrops"

#
# Include typehints in descriptions
autodoc_typehints = "description"


# Both the class' and the __init__ method's docstring are concatenated and inserted.
autoclass_content = "both"

# Content is in the same order as in module
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["**.ipynb_checkpoints", "custom_reference*"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/j-desloires/eo-crops",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "develop",
    "path_to_docs": "docs",
    "use_download_button": True,
    "extra_navbar": "",
}

html_logo = "../figures/logo.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "eocropsdoc"
# show/hide links for source
html_show_sourcelink = False

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#    (master_doc, "eo-learn.tex", doc_title, author, "manual"),
# ]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "eocrops", doc_title, [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "eocrops",
        doc_title,
        author,
        "eocrops",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/3.8/": None}


# -- Custom settings ----------------------------------------------

# When Sphinx documents class signature it prioritizes __new__ method over __init__ method. The following hack puts

sphinx.ext.autodoc._CLASS_NEW_BLACKLIST.append(
    "{0.__module__}.{0.__qualname__}".format(eocrops.__new__)
)  # noqa[SLF001]


EXAMPLES_FOLDER = "./eocrops/examples"
MARKDOWNS_FOLDER = "./markdowns"


def copy_documentation_examples(source_folder, target_folder):
    """Makes sure to copy only notebooks that are actually included in the documentation"""
    files_to_include = []
    for rst_file in ["eocrops/example.rst"]:
        content = pathlib.Path(rst_file).read_text()
        for line in content.split("\n"):
            line = line.strip(" \t")
            if line.startswith("examples/"):
                files_to_include.append(line.split("/", 1)[1])

    for file in files_to_include:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copyfile(source_path, target_path)


# copy examples
shutil.rmtree(EXAMPLES_FOLDER, ignore_errors=True)
copy_documentation_examples(
    os.path.join(os.path.dirname(__file__), "../../examples"),
    os.path.join(os.path.dirname(__file__), EXAMPLES_FOLDER),
)

shutil.copyfile("../../README.md", os.path.join(MARKDOWNS_FOLDER, "README.md"))
shutil.rmtree(os.path.join(MARKDOWNS_FOLDER, "docs/figures"), ignore_errors=True)
shutil.copytree("../../docs/figures", os.path.join(MARKDOWNS_FOLDER, "docs/figures"))


# Auto-generate documentation pages
current_dir = os.path.abspath(os.path.dirname(__file__))
repository_dir = os.path.join(current_dir, "..", "..")
modules = ["inputs", "tasks", "utils"]

APIDOC_OPTIONS = [
    "--module-first",
    "--separate",
    "--no-toc",
    "--templatedir",
    os.path.join(current_dir, "_templates"),
]

autodoc_default_options = {"inherited-members": False}


def is_error(obj):
    return issubclass(obj, Exception)


conditionally_ignored = {
    "__reduce__": is_error,
    "__init__": is_error,
    "with_traceback": is_error,
    "args": is_error,
}


def skip_member_handler(app, objtype, membername, member, skip, options):
    """Avoid display methods from parent class"""
    ignore_checker = conditionally_ignored.get(membername)
    if ignore_checker:
        frame = sys._getframe()
        while frame.f_code.co_name != "filter_members":
            frame = frame.f_back

        suspect = frame.f_locals["self"].object
        return not ignore_checker(suspect)
    return skip


def _html_page_context(app, pagename, templatename, context, doctree):
    # Disable edit button for docstring generated pages
    if "generated" in pagename:
        context["theme_use_edit_page_button"] = False


def setup(app):
    """dummy docstring for pydocstyle"""
    app.connect("html-page-context", _html_page_context)
    app.connect("autodoc-skip-member", skip_member_handler)
