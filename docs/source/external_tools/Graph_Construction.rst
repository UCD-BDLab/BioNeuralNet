Graph Construction
==================

These wrappers facilitate calling **R-based** adjacency-building approaches:

**SmCCNet**:
  - Constructs networks via sparse canonical correlation. Ideal for multi-omics correlation or partial correlation tasks.

.. literalinclude:: ../examples/smccnet_example.py
   :language: python
   :caption: Using SmCCNet to build an adjacency matrix from omics + phenotype data.

**WGCNA**:
  - Weighted Gene Co-expression Network Analysis wrapper for R’s WGCNA package.

.. literalinclude:: ../examples/wgcna_example.py
   :language: python
   :caption: Demonstration of WGCNA adjacency generation from expression data.

**Note**:
1. You must have R installed, plus the respective CRAN packages (e.g. “WGCNA” or “SmCCNet”), for these wrappers to work.
2. The adjacency matrices generated here can then be passed to GNNEmbedding, DPMON, or other BioNeuralNet modules.
