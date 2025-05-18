Graph Construction
==================

These wrappers facilitate calling **R-based** adjacency-building approaches:

**SmCCNet**:
  - Constructs networks via sparse canonical correlation. Ideal for multi-omics correlation or partial correlation tasks.

.. literalinclude:: ../examples/smccnet_example.py
   :language: python
   :caption: Using SmCCNet to build an adjacency matrix from omics + phenotype data.

**Note**:
  
  - You must have R installed, plus the respective CRAN packages  (“SmCCNet”), for these wrappers to work.
  - The adjacency matrices generated here can then be passed to GNNEmbedding, DPMON, or other BioNeuralNet modules.
