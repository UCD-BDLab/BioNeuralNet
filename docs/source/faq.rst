Frequently Asked Questions (FAQ)
================================

**Q1: What is BioNeuralNet?**

A1: BioNeuralNet is a Python-based framework that integrates **multi-omics** data with **Graph Neural Networks (GNNs)**
for clustering, embedding, and disease prediction. It converts high-dimensional omics networks into lower-dimensional
representations, enabling downstream tasks like subject representation, subnetwork discovery, and phenotype prediction.

**Q2: What are the key components of BioNeuralNet?**

A2: BioNeuralNet covers five main steps:

1. **Graph Construction**: (External) Build adjacency matrices using WGCNA, SmCCNet, or your own methods.
2. **Graph Clustering**: Identify modules via hierarchical clustering, PageRank, or Louvain.
3. **Network Embedding**: Generate embeddings with GNNs or Node2Vec.
4. **Subject Representation**: Integrate node embeddings into omics data for enhanced feature sets.
5. **Downstream Tasks**: E.g., disease prediction using **DPMON**, an end-to-end pipeline that trains GNN + NN.

**Q3: How do I install BioNeuralNet and dependencies?**

A3: Simply do:

.. code-block:: bash

   pip install bioneuralnet==0.1.0b1

Then install PyTorch separately from [PyTorch.org](https://pytorch.org/get-started/locally/).
Optionally install R-based packages (WGCNA, SmCCNet) if you want to construct adjacency matrices with them.
See :doc:`installation` for more details.

**Q4: Does BioNeuralNet support GPU acceleration?**

A4: Yes. Install a GPU-compatible version of PyTorch (with CUDA) from the PyTorch site.
BioNeuralNet will automatically detect and use CUDA if `torch.cuda.is_available()` returns True.

**Q5: Can I contribute custom components (e.g., new GNN architectures)?**

A5: Definitely! We welcome contributions. Fork the repository, add your module or architecture,
and submit a pull request. Check the `README <https://github.com/UCD-BDLab/BioNeuralNet/blob/main/README.md>`_
and see instructions on code style, testing, and docstrings.

**Q6: How does DPMON differ from a standard GNN approach?**

A6: **DPMON** is an end-to-end pipeline that trains both a GNN (for node embeddings) and a neural
network classifier jointly, focusing on **local + global** structure for disease phenotypes. It does
*not* rely on correlation-based node labeling; rather, it fuses multi-omics data with the adjacency
matrix to learn disease-predictive embeddings in a single integrated model.

**Q7: Can BioNeuralNet work with unsupervised GNN embeddings?**

A7: Yes. If you do not provide a numeric label for each node, the GNN can run unsupervised (via
basic adjacency-based features). For truly self-supervised tasks (e.g., contrastive learning),
you might adapt code or rely on external libraries. But you can definitely skip explicit labels
and still get a structural embedding.

**Q8: Where are the advanced clustering or visualization methods found?**

A8: We provide classes like `PageRank`, `HierarchicalClustering`, `StaticVisualizer`,
and `DynamicVisualizer` under `bioneuralnet.external_tools` or within the pipeline’s
tool suite. These methods are separate from the core embedding logic but integrate seamlessly
once you have adjacency matrices or embedded data.

**Q9: I have existing adjacency matrices. Do I need WGCNA or SmCCNet?**

A9: Not necessarily. If your adjacency matrix is already built, you can pass it directly to
GNNEmbedding or DPMON. The external tool wrappers (WGCNA, SmCCNet) are optional conveniences
for generating adjacency matrices in R.

**Q10: What license is BioNeuralNet released under?**

A10: BioNeuralNet is under the `MIT License <https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE>`_.

**Q11: How do I report issues or request features?**

A11: Please open an Issue on our GitHub repository `UCD-BDLab/BioNeuralNet <https://github.com/UCD-BDLab/BioNeuralNet/issues>`_.
We track bugs, enhancements, and feature requests there.

**Q12: Where can I find more examples or tutorials?**

A12: Check out :doc:`tutorials/index` for step-by-step workflows demonstrating
graph construction, embedding generation, subject representation, and disease prediction.
