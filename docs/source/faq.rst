Frequently Asked Questions (FAQ)
================================

**Q1: What is BioNeuralNet?**

A1: BioNeuralNet is a Python framework for integrating **multi-omics data** with **Graph Neural Networks (GNNs)**.  
It provides tools for **graph construction, clustering, network embedding, subject representation, and disease prediction**.

**Q2: What are the key features of BioNeuralNet?**

A2: BioNeuralNet includes the following components:

- **Graph Construction**: Build multi-omics networks using **SmCCNet** or custom adjacency matrices.
- **Graph Clustering**: Identify meaningful communities with **Louvain, Hybrid Louvain, or PageRank-based methods**.
- **GNN Embedding**: Learn **node embeddings** from biological graphs with `GNNEmbedding`.
- **Subject Representation**: Integrate embeddings into omics data via `GraphEmbedding` for enhanced feature learning.
- **Disease Prediction**: Use **DPMON**, an end-to-end pipeline that trains a GNN-based classifier.

**Q3: How do I install BioNeuralNet?**

A3: Install BioNeuralNet with:

.. code-block:: bash

   pip install bioneuralnet

For GPU support, install PyTorch with CUDA from [PyTorch.org](https://pytorch.org/get-started/locally/).  
See :doc:`installation` for full setup details.

**Q4: Does BioNeuralNet support GPU acceleration?**

A4: Yes. If you have a **CUDA-compatible GPU**, BioNeuralNet will automatically use it if `torch.cuda.is_available()` is `True`.

**Q5: Can I use my own adjacency matrix instead of SmCCNet?**

A5: Yes! If you have a **precomputed adjacency matrix**, you can pass it directly to `GNNEmbedding` or `DPMON`.  
SmCCNet is an optional tool for generating adjacency matrices.

**Q6: How is DPMON different from other GNN models?**

A6: **DPMON** is designed for **multi-omics disease prediction**. Unlike standard GNNs, it:  
   - Jointly learns **node embeddings and a classifier**.  
   - Leverages both **local and global graph structures**.  
   - Integrates **phenotype and clinical data** alongside omics features.

**Q7: Can I run GNNEmbedding without labeled data (unsupervised learning)?**

A7: Yes! If you don't provide labels, **GNNEmbedding** will still generate embeddings based on graph structure.  
For **self-supervised learning** (e.g., contrastive learning), you may need additional adaptation.

**Q8: What clustering methods does BioNeuralNet support?**

A8: BioNeuralNet provides:  
   - **Correlated Louvain**: Clusters nodes based on omics similarity and phenotype correlation.  
   - **Hybrid Louvain**: Iteratively refines clusters using **PageRank expansion**.  
   - **Correlated PageRank**: Detects communities based on personalized PageRank scores.

**Q9: Can I contribute new features or models?**

A9: Yes! We welcome contributions. Fork the repository, add your module, and submit a pull request.  
Check our `contribution guide <https://github.com/UCD-BDLab/BioNeuralNet/blob/main/README.md>`_.

**Q10: What license is BioNeuralNet under?**

A10: BioNeuralNet is released under the `MIT License <https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE>`_.

**Q11: How do I report issues or request features?**

A11: Open an issue on our GitHub repository:  
`UCD-BDLab/BioNeuralNet <https://github.com/UCD-BDLab/BioNeuralNet/issues>`_.

**Q12: Where can I find tutorials or example scripts?**

A12: See :doc:`tutorials/index` for **step-by-step guides** on graph construction, embeddings, subject representation, and disease prediction.
