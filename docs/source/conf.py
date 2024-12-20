import os
import sys
import pkg_resources
sys.path.insert(0, os.path.abspath('../../'))

project = 'BioNeuralNet'
author = 'Vicente Ramos'
release = pkg_resources.get_distribution("bioneuralnet").version

extensions = [
    'sphinx.ext.autodoc',      
    'sphinx.ext.napoleon',     
    'sphinx.ext.viewcode',     
    'sphinx.ext.autosummary',  
    'sphinx.ext.intersphinx', 
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'  
html_static_path = ['_static']
autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
}
