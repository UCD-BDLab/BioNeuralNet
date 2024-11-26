# docs/source/conf.py

import os
import sys
import pkg_resources

# -- Path setup --------------------------------------------------------------

# Add the project directory to sys.path
# Adjust the path as necessary based on your project structure
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'BioNeuralNet'
author = 'Vicente Ramos'
release = pkg_resources.get_distribution("bioneuralnet").version
ß
# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',      # Automatically document modules
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx.ext.intersphinx',  # Link to other project's documentation
]

# Napoleon settings for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Templates path
templates_path = ['_templates']

# List of patterns to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'  # Read the Docs theme

# Add any paths that contain custom static files here
html_static_path = ['_static']

# -- Autosummary configuration -----------------------------------------------

autosummary_generate = True

# -- Intersphinx configuration ------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    # Add more mappings as needed
}
