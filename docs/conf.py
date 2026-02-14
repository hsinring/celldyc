# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
sys.path.insert(0, os.path.abspath('..'))

add_module_names = False

project = 'CellDyc'
copyright = '2026, Hsinring. All rights reserved'
author = 'Hsinring, Xin'
# release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_nb',
    "sphinx_design",
    'sphinx.ext.autosummary'
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': -1,
    'includehidden': True,
    'titles_only': False,
    'includehidden': True
}
html_sidebars = {
    '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'],
}
html_css_files = [
    'custom.css',
]
html_static_path = ['_static']

autodoc_default_options = {
    'members': True,          
    'member-order': 'bysource',
    'show-inheritance': True,  
    'exclude-members': '__weakref__'
}
nb_execution_mode = "off"


