Acknowledgments
===============

We gratefully acknowledge the TOPMed consortium for providing critical datasets, and thank our collaborators for their valuable contributions. This work was supported in part by the Graduate Assistance in Areas of National Need (GAANN) Fellowship, funded by the U.S. Department of Education.

Key Dependencies
----------------

BioNeuralNet integrates multiple open-source libraries to deliver advanced multi-omics integration and analysis. We acknowledge the following key dependencies:

- **PyTorch:** Deep learning and GNN computations. `PyTorch <https://github.com/pytorch/pytorch/>`_
- **PyTorch Geometric:** Efficient graph neural network implementations. `PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric/>`_
- **NetworkX:** Robust graph data structures and algorithms. `NetworkX <https://github.com/networkx/networkx/>`_
- **Scikit-learn:** Dimensionality reduction via PCA and accuracy metrics.  `Scikit-learn <https://github.com/scikit-learn/scikit-learn/>`_
- **pandas:** Core data manipulation and analysis tools. `pandas <https://github.com/pandas-dev/pandas/>`_
- **numpy:** Fundamental package for scientific computing. `numpy <https://github.com/numpy/numpy/>`_
- **ray[tune]:** Scalable hyperparameter tuning for GNN models. `ray[tune] <https://docs.ray.io/en/latest/tune/>`_
- **matplotlib:** Data visualization. `matplotlib <https://github.com/matplotlib/matplotlib/>`_
- **python-louvain:** Community detection algorithms for graphs. `python louvain <https://github.com/taynaud/python-louvain/>`_

We also acknowledge R-based tools for external network construction:

- **SmCCNet** - Sparse multiple canonical correlation network tool. `SmCCNet <https://cran.r-project.org/web/packages/SmCCNet/>`_
- **WGCNA** - Weighted gene co-expression network analysis. `WGCNA <https://cran.r-project.org/web/packages/WGCNA/>`_

These tools enhance BioNeuralNet's capabilities without being required for its core functionality.

Contributors
------------
Contributions to BioNeuralNet are welcome. If you wish to contribute new features, report issues, or provide feedback, please visit our GitHub repository:

`UCD-BDLab/BioNeuralNet <https://github.com/UCD-BDLab/BioNeuralNet>`_

Please refer to our contribution guidelines in the repository for more details.

Frequently Asked Questions (FAQ)
--------------------------------

**Q1: What is BioNeuralNet?**  
A1: BioNeuralNet is a Python framework for integrating multi-omics data with Graph Neural Networks (GNNs). It provides end-to-end solutions for network embedding, clustering, subject representation, and disease prediction.

**Q2: What are the key features of BioNeuralNet?**  
A2:  
- **Graph Clustering:** Identify communities using Correlated Louvain, Hybrid Louvain, and Correlated PageRank methods.  
- **GNN Embedding:** Generate node embeddings using advanced GNN models.  
- **Subject Representation:** Enrich omics data with learned embeddings.  
- **Disease Prediction:** Leverage DPMON for integrated, end-to-end disease prediction.

**Q3: How do I install BioNeuralNet?**  
A3: Install via pip:

.. code-block:: bash

   pip install bioneuralnet

For full installation instructions, see the :doc:`installation` guide.

**Q4: Does BioNeuralNet support GPU acceleration?**  
A4: Yes. If a CUDA-compatible GPU is available, BioNeuralNet will utilize it via PyTorch.

**Q5: Can I use my own network instead of SmCCNet?**  
A5: Absolutely. You can supply a pre-computed adjacency matrix directly to the GNNEmbedding or DPMON modules.

**Q6: How is DPMON different from standard GNN models?**  
A6: DPMON is tailored for multi-omics disease prediction by jointly learning node embeddings and a classifier, integrating both local and global graph structures.

**Q7: What clustering methods does BioNeuralNet support?**  
A7: BioNeuralNet offers:  
- Correlated Louvain  
- Hybrid Louvain  
- Correlated PageRank

**Q8: How can I contribute to BioNeuralNet?**  
A8: Contributions are encouraged! Fork the repository, develop your feature, and submit a pull request. See our contribution guidelines on GitHub.

**Q9: Where can I find tutorials and examples?**  
A9: For detailed guides and demos, visit :doc:`tutorials/index` and check out the example notebooks provided in the repository.

**Q10: What license is BioNeuralNet released under?**  
A10: BioNeuralNet is distributed under the MIT License. For details, see the `MIT LICENSE <https://github.com/UCD-BDLab/BioNeuralNet?tab=MIT-1-ov-file>`_ page.

Return to :doc:`../index`
