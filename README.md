# BioNeuralNet: A Graph Neural Network based Multi-Omics Network Data Analysis Tool

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bioneuralnet)](https://pypi.org/project/bioneuralnet/)
[![GitHub Issues](https://img.shields.io/github/issues/UCD-BDLab/BioNeuralNet)](https://github.com/UCD-BDLab/BioNeuralNet/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/UCD-BDLab/BioNeuralNet)](https://github.com/UCD-BDLab/BioNeuralNet/graphs/contributors)
[![Downloads](https://static.pepy.tech/badge/bioneuralnet)](https://pepy.tech/project/bioneuralnet)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue.svg)](https://bioneuralnet.readthedocs.io/en/latest/)


## Welcome to BioNeuralNet 1.0.9

![BioNeuralNet Logo](assets/LOGO_WB.png)

**BioNeuralNet** is a Python framework for integrating and analyzing multi-omics data using **Graph Neural Networks (GNNs)**.
It provides tools for network construction, embedding generation, clustering, and disease prediction, all within a modular, scalable, and reproducible pipeline.

![BioNeuralNet Workflow](assets/BioNeuralNet.png)

## Documentation

**[BioNeuralNet Documentation & Examples](https://bioneuralnet.readthedocs.io/en/latest/)**

## Table of Contents

- [1. Installation](#1-installation)
  - [1.1. Install BioNeuralNet](#11-install-bioneuralnet)
  - [1.2. Install PyTorch and PyTorch Geometric](#12-install-pytorch-and-pytorch-geometric)
- [2. BioNeuralNet Core Features](#2-bioneuralnet-core-features)
- [3. Quick Example: SmCCNet + DPMON for Disease Prediction](#3-quick-example-smccnet--dpmon-for-disease-prediction)
- [4. Documentation and Tutorials](#4-documentation-and-tutorials)
- [5. Frequently Asked Questions (FAQ)](#5-frequently-asked-questions-faq)
- [6. Acknowledgments](#6-acknowledgments)
- [7. Testing and Continuous Integration](#7-testing-and-continuous-integration)
- [8. Contributing](#8-contributing)
- [9. License](#9-license)
- [10. Contact](#10-contact)
- [11. References](#11-References)

## 1. Installation

BioNeuralNet supports Python `3.10`, `3.11` and `3.12`.

### 1.1. Install BioNeuralNet
```bash
pip install bioneuralnet
```

## 1.2. Install PyTorch and PyTorch Geometric
BioNeuralNet relies on PyTorch for GNN computations. Install PyTorch separately:

- **PyTorch (CPU)**:
  ```bash
  pip install torch torchvision torchaudio
  ```

- **PyTorch Geometric**:
  ```bash
  pip install torch_geometric
  ```

For GPU acceleration, please refer to:
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


## **BioNeuralNet Core Features**

For an end-to-end example of BioNeuralNet, see the [Quick Start](https://bioneuralnet.readthedocs.io/en/latest/Quick_Start.html) and [TCGA-BRCA Dataset](https://bioneuralnet.readthedocs.io/en/latest/TCGA-BRCA_Dataset.html) guides.

### [Network Embedding](https://bioneuralnet.readthedocs.io/en/latest/gnns.html)
- Given a multi-omics network as input, BioNeuralNet can generate embeddings using Graph Neural Networks (GNNs).
- Generate embeddings using methods such as **GCN**, **GAT**, **GraphSAGE**, and **GIN**.
- Outputs can be obtained as native tensors or converted to pandas DataFrames for easy analysis and visualization.
- Embeddings unlock numerous downstream applications, including disease prediction, enhanced subject representation, clustering, and more.

### [Graph Clustering](https://bioneuralnet.readthedocs.io/en/latest/clustering.html)
- Identify functional modules or communities using **correlated clustering methods** (e.g., `CorrelatedPageRank`, `CorrelatedLouvain`, `HybridLouvain`) that integrate phenotype correlation to extract biologically relevant modules [[1]](#1).
- Clustering methods can be applied to any network representation, allowing flexible analysis across different domains.
- All clustering components return either raw partition dictionaries or induced subnetwork adjacency matrices (as DataFrames) for visualization.
- Use cases include feature selection, biomarker discovery, and network-based analysis.

### [Downstream Tasks](https://bioneuralnet.readthedocs.io/en/latest/downstream_tasks.html)

#### Subject Representation
- Integrate node embeddings back into omics data to enrich subject-level profiles by weighting features with the learned embedding.
- This embedding-enriched data can be used for downstream tasks such as disease prediction or biomarker discovery.
- The result can be returned as a DataFrame or a PyTorch tensor, fitting naturally into downstream analyses.

#### Disease Prediction for Multi-Omics Network (DPMON) [[2]](#2)
- Classification end-to-end pipeline for disease prediction using Graph Neural Network embeddings.
- DPMON supports hyperparameter tuning, when enabled, it finds the best configuration for the given data.
- This approach, along with native pandas integration across modules, ensures that BioNeuralNet can be easily incorporated into your analysis workflows.

### [Metrics](https://bioneuralnet.readthedocs.io/en/latest/metrics.html)
- Visualize embeddings, feature variance, clustering comparison, and network structure in 2D.
- Evaluate embedding quality and clustering relevance using correlation with phenotype.
- Performance benchmarking tools for classification tasks using various models.
- Useful for assessing feature importance, validating network structure, and comparing cluster outputs.

### [Utilities](https://bioneuralnet.readthedocs.io/en/latest/utils.html)
- Build graphs using k-NN similarity, Pearson/Spearman correlation, RBF kernels, mutual information, or soft-thresholding.
- Filter and preprocess omics or clinical data by variance, correlation, random forest importance, or ANOVA F-test.
- Tools for network pruning, feature selection, and data cleaning.
- Quickly summarize datasets with variance, zero-fraction, expression level, or correlation overviews.
- Includes conversion tools for RData and integrated logging.

### [External Tools](https://bioneuralnet.readthedocs.io/en/latest/external_tools/index.html)
- **Graph Construction**:
  - BioNeuralNet provides additional tools in the `bioneuralnet.external_tools` module.
  - Includes support for **SmCCNet** (Sparse Multiple Canonical Correlation Network), an R-based tool for constructing phenotype-informed correlation networks [[3]](#3).
  - These tools are optional but enhance BioNeuralNet’s graph construction capabilities and are recommended for more integrative or exploratory workflows.


## 3. Example: SmCCNet + DPMON for Disease Prediction

```python
import pandas as pd
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.downstream_task import DPMON
from bioneuralnet.datasets import DatasetLoader

# Step 1: Load your data or use one of the provided datasets
Example = DatasetLoader("example1")
omics_proteins = Example.data["X1"]
omics_metabolites = Example.data["X2"]
phenotype_data = Example.data["Y"]
clinical_data = Example.data["clinical_data"]

# Step 2: Network Construction
smccnet = SmCCNet(
    phenotype_df=phenotype_data,
    omics_dfs=[omics_proteins, omics_metabolites],
    data_types=["protein", "metabolite"],
    kfold=5,
    summarization="PCA",
)
global_network, clusters = smccnet.run()
print("Adjacency matrix generated.")

# Step 3: Disease Prediction (DPMON)
dpmon = DPMON(
    adjacency_matrix=global_network,
    omics_list=[omics_proteins, omics_metabolites],
    phenotype_data=phenotype_data,
    clinical_data=clinical_data,
    model="GCN",
)
predictions = dpmon.run()
print("Disease phenotype predictions:\n", predictions)
```

## 4. Documentation and Tutorials

- **Full documentation**: [BioNeuralNet Documentation](https://bioneuralnet.readthedocs.io/en/latest/)

- **Jupyter Notebook Examples**:
  - [Quick Start](https://bioneuralnet.readthedocs.io/en/latest/Quick_Start.html)
  - [TCGA-BRCA Dataset](https://bioneuralnet.readthedocs.io/en/latest/TCGA-BRCA_Dataset.html)

- Tutorials include:
  - Multi-omics graph construction.
  - GNN embeddings for disease prediction.
  - Subject representation with integrated embeddings.
  - Clustering using Hybrid Louvain and Correlated PageRank.
- API details are available in the [API Reference](https://bioneuralnet.readthedocs.io/en/latest/api.html).

## 5. Frequently Asked Questions (FAQ)

- **Does BioNeuralNet support GPU acceleration?**
  Yes, install PyTorch with CUDA support.

- **Can I use my own omics network?**
  Yes, you can provide a custom network as an adjancy matrix instead of using SmCCNet.

- **What clustering methods are supported?**
  BioNeuralNet supports Correlated Louvain, Hybrid Louvain, and Correlated PageRank.

For more FAQs, please visit our [FAQ page](https://bioneuralnet.readthedocs.io/en/latest/faq.html).

## 6. Acknowledgments

BioNeuralNet integrates multiple open-source libraries. We acknowledge key dependencies:

- [**PyTorch**](https://github.com/pytorch/pytorch) - GNN computations and deep learning models.
- [**PyTorch Geometric**](https://github.com/pyg-team/pytorch_geometric) - Graph-based learning for multi-omics.
- [**NetworkX**](https://github.com/networkx/networkx) - Graph data structures and algorithms.
- [**Scikit-learn**](https://github.com/scikit-learn/scikit-learn) - Feature selection and evaluation utilities.
- [**pandas**](https://github.com/pandas-dev/pandas) & [**numpy**](https://github.com/numpy/numpy) - Core data processing tools.
- [**ray[tune]**](https://github.com/ray-project/ray) - Hyperparameter tuning for GNN models.
- [**matplotlib**](https://github.com/matplotlib/matplotlib) - Data visualization.
- [**cptac**](https://github.com/PNNL-CompBio/cptac) - Dataset handling for clinical proteomics.
- [**python-louvain**](https://github.com/taynaud/python-louvain) - Community detection algorithms.
- [**statsmodels**](https://github.com/statsmodels/statsmodels) - Statistical models and hypothesis testing (e.g., ANOVA, regression).

We also acknowledge R-based tools for external network construction:

- [**SmCCNet**](https://github.com/UCD-BDLab/BioNeuralNet/tree/main/bioneuralnet/external_tools/smccnet) - Sparse multiple canonical correlation network.

## 7. Testing and Continuous Integration

- **Run Tests Locally:**
   ```bash
   pytest --cov=bioneuralnet --cov-report=html
   open htmlcov/index.html
   ```

- **Continuous Integration:**
   GitHub Actions runs automated tests on every commit.

## 8. Contributing

We welcome contributions! To get started:

```bash
git clone https://github.com/UCD-BDLab/BioNeuralNet.git
cd BioNeuralNet
pip install -r requirements-dev.txt
pre-commit install
pytest
```

### How to Contribute

- Fork the repository, create a new branch, and implement your changes.
- Add tests and documentation for any new features.
- Submit a pull request with a clear description of your changes.

## 9. License

BioNeuralNet is distributed under the [MIT License](https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE).

## 10. Contact

- **Issues and Feature Requests:** [Open an Issue](https://github.com/UCD-BDLab/BioNeuralNet/issues)
- **Email:** [vicente.ramos@ucdenver.edu](mailto:vicente.ramos@ucdenver.edu)

## 11. References

<a id="1">[1]</a> Abdel-Hafiz, M., Najafi, M., et al. "Significant Subgraph Detection in Multi-omics Networks for Disease Pathway Identification." *Frontiers in Big Data*, 5 (2022). [DOI: 10.3389/fdata.2022.894632](https://doi.org/10.3389/fdata.2022.894632)

<a id="2">[2]</a> Hussein, S., Ramos, V., et al. "Learning from Multi-Omics Networks to Enhance Disease Prediction: An Optimized Network Embedding and Fusion Approach." In *2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, Lisbon, Portugal, 2024, pp. 4371-4378. [DOI: 10.1109/BIBM62325.2024.10822233](https://doi.org/10.1109/BIBM62325.2024.10822233)

<a id="3">[3]</a> Liu, W., Vu, T., Konigsberg, I. R., Pratte, K. A., Zhuang, Y., & Kechris, K. J. (2023). "Network-Based Integration of Multi-Omics Data for Biomarker Discovery and Phenotype Prediction." *Bioinformatics*, 39(5), btat204. [DOI: 10.1093/bioinformatics/btat204](https://doi.org/10.1093/bioinformatics/btat204)
