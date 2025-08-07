import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath("../../"))

try:
    release = metadata.version("bioneuralnet")
except metadata.PackageNotFoundError:
    release = "1.1.1"

project = "BioNeuralNet"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "myst_nb",
]
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence"]

pygments_style = "monokai"
pygments_dark_style = "monokai"
highlight_language = "python"
nbsphinx_codecell_lexer = "ipython3"
nbsphinx_execute = "never"
nbsphinx_prompt_width = "0em"

autosummary_imported_members = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme = "furo"
#html_title = ""
html_logo = "_static/LOGO_TB.svg"
#html_theme = "sphinx_rtd_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}

autodoc_mock_imports = [
    "torch",
    "torch_geometric",
]
