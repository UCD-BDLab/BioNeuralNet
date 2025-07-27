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
- **statsmodels:** Statistical tests and models, including ANOVA and linear regression. `statsmodels <https://github.com/statsmodels/statsmodels/>`_

We also acknowledge R-based tools for external network construction:

- **SmCCNet**: Sparse multiple canonical correlation network tool. `SmCCNet <https://cran.r-project.org/web/packages/SmCCNet/>`_

These tools enhance BioNeuralNet's capabilities without being required for its core functionality.

Contributors
------------
Contributions to BioNeuralNet are welcome. If you wish to contribute new features, report issues, or provide feedback, please visit our GitHub repository:

`UCD-BDLab/BioNeuralNet <https://github.com/UCD-BDLab/BioNeuralNet>`_

   - **Ways to contribute**:
   
      - Report issues or bugs on our `GitHub Issues page <https://github.com/UCD-BDLab/BioNeuralNet/issues>`_.
      - Suggest new features or improvements.
      - Share your experiences or use cases with the community.

   - **Implementing new features**:

      - Fork the repo and create a feature branch.
      - Add tests and documentation for new features.
      - Run the test suite and and pre-commit hooks before opening a Pull Request(PR).
      - A new PR should pass all tests and adhere to the project's coding standards.

.. code-block:: bash
   
   git clone https://github.com/UCD-BDLab/BioNeuralNet.git
   cd BioNeuralNet
   pip install -r requirements-dev.txt
   pre-commit install
   pytest --cov=bioneuralnet

Frequently Asked Questions (FAQ)
--------------------------------

**Q1: What is BioNeuralNet?**:

   - BioNeuralNet is a flexible, modular Python framework developed to facilitate end-to-end network-based multi-omics analysis** using **Graph Neural Networks (GNNs)**. It addresses the complexities associated with multi-omics data, such as high dimensionality, sparsity, and intricate molecular interactions, by converting biological networks into meaningful, low-dimensional embeddings suitable for downstream tasks.

**Q2: What are the key features of BioNeuralNet?**:

   - **Graph Clustering:** Identify communities using Correlated Louvain, Hybrid Louvain, and Correlated PageRank methods.  
   - **GNN Embedding:** Generate node embeddings using advanced GNN models.  
   - **Subject Representation:** Enrich omics data with learned embeddings.  
   - **Disease Prediction:** Leverage DPMON for integrated, end-to-end disease prediction.

**Q3: How do I install BioNeuralNet?**:

.. code-block:: bash

   pip install bioneuralnet


**Q4: Does BioNeuralNet support GPU acceleration?**:

   - Yes. If a CUDA-compatible GPU is available, BioNeuralNet will utilize it via PyTorch.

**Q5: Can I use my own network instead of SmCCNet or internal graph generation functions?**

   - Absolutely. You can supply a pre-computed adjacency matrix directly to the **GNNEmbedding** or **DPMON** modules.

**Q6: Where can I find tutorials and examples?**:

   - We provide a set of tutorials and example notebooks to help you get started with BioNeuralNet. You can find them in the `tutorials` directory of the repository.  
   - For a quick start, check out the following notebooks:

      - :doc:`Quick_Start`.
      - :doc:`TCGA-BRCA_Dataset`.

**Q7: What license is BioNeuralNet released under?**:

   - BioNeuralNet is distributed under the `Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0) <https://creativecommons.org/licenses/by-nc-nd/4.0/>`_.

Return to :doc:`../index`
