External Tools
==============

BioNeuralNet provides utility functions for interoperability between Python and R, allowing users to export cross-validation folds and network matrices from R-based tools and load them directly into BioNeuralNet workflows.

Available functions:

- ``extract_and_load_folds``: Triggers R script extraction and loads CV folds.
- ``load_r_export_folds``: Loads a previously extracted R directory structure.
- ``rdata_to_df``: Converts an ``.RData`` file object to a pandas DataFrame.

.. note::

   R must be installed and available through system path. Adjacency matrices generated from R can be passed directly to ``NetworkAnalyzer``, ``DPMON``, or other BioNeuralNet modules.

.. toctree::
   :maxdepth: 2
   :caption: External Tools