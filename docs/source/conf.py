import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath("../../"))

try:
    release = metadata.version("bioneuralnet")
except metadata.PackageNotFoundError:
    release = "1.0"

project = "BioNeuralNet"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

pygments_style = "sphinx"
highlight_language = "python"
nbsphinx_codecell_lexer = "ipython3"

autosummary_imported_members = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "bioneuralnet.external_tools",
    "bioneuralnet.external_tools.node2vec",
]
