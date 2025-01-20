Subject Representation
======================

**SubjectRepresentation** (GraphEmbedding) enriches omics data by incorporating node embeddings:

.. literalinclude:: ../examples/subject_representation_example.py
   :language: python
   :caption: Example of Integrating Node Embeddings into Subject-Level Data

---------------------------------------

Disease Prediction (DPMON)
==========================

**DPMON** trains a GNN + neural network classifier end-to-end for disease phenotypes:

.. literalinclude:: ../examples/dpmon_example.py
   :language: python
   :caption: Example of Running DPMON with adjacency + omics data

Key points:
- **Local + Global** graph structure
- End-to-end optimization of embeddings + classifier
- Minimizes risk of overfitting by factoring in network connectivity
