Graph Construction
==================

These wrappers facilitate calling **R-based** packages for network construction, specifically **SmCCNet**.:

**SmCCNet**:

  - Constructs networks via sparse canonical correlation. Ideal for multi-omics correlation or partial correlation tasks.

**Note**:
  
  - You must have R installed, plus the respective CRAN packages  (“SmCCNet”), for these wrappers to work.
  - The adjacency matrices generated here can then be passed to GNNEmbedding, DPMON, or other BioNeuralNet modules.
