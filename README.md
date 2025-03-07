# BioNeuralNet: Multi-Omics Integration with Graph Neural Networks

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyPI](https://img.shields.io/pypi/v/bioneuralnet)
![GitHub Issues](https://img.shields.io/github/issues/UCD-BDLab/BioNeuralNet)
![GitHub Contributors](https://img.shields.io/github/contributors/UCD-BDLab/BioNeuralNet)
![Downloads](https://static.pepy.tech/badge/bioneuralnet)

[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue.svg)](https://bioneuralnet.readthedocs.io/en/latest/)

## Welcome to BioNeuralNet 1.0

![BioNeuralNet Logo](assets/LOGO_WB.png)

BioNeuralNet is a robust Python framework for integrating multi-omics data with Graph Neural Networks (GNNs).

![BioNeuralNet Workflow](assets/BioNeuralNet.png)

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

## 1. Installation

BioNeuralNet supports Python 3.10 and 3.11.

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


## 2. BioNeuralNet Core Features

For an End-to-End example example of BioNeuralNet, see [BioNeuralNet Demo](https://bioneuralnet.readthedocs.io/en/latest/BioNeuralNet.html)

**Network Embedding**:

- Given a multi-omics network as input, BioNeuralNet can generate embeddings using Graph Neural Networks (GNNs).
- Generate embeddings using methods such as **GCN**, **GAT**, **GraphSAGE**, and **GIN**.
- Outputs can be obtained as native tensors or converted to pandas DataFrames for easy analysis and visualization.
- Embeddings unlock numerous downstream applications, including disease prediction, enhanced subject representation, clustering, and more.

**Graph Clustering**:

- Identify functional modules or communities using **correlated clustering methods** (e.g., CorrelatedPageRank, CorrelatedLouvain, HybridLouvain) that integrate phenotype correlation to extract biologically relevant modules [1]_.
- Clustering methods can be applied to any network represented allowing flexible analysis across different domains.
- All clustering components return either raw partitions dictionaries or induced subnetwork adjacency matrices (as DataFrames) for visualization.
- Use cases include, feature selection, biomarker discovery, and network-based analysis.

**Downstream Tasks**:

- **Subject Representation**:

   - Integrate node embeddings back into omics data to enrich subject-level profiles by weighting features with learned embedding.
   - This embedding-enriched data can be used for downstream tasks such as disease prediction or biomarker discovery.
   - The result can be returned as a DataFrame or a PyTorch tensor, fitting naturally into downstream analyses.

- **Disease Prediction for Multi-Omics Network DPMON**:

   - Classification End-to-End pipeline for disease prediction using Graph Neural Network embeddings.
   - DPMON supports hyperparameter tuning-when enabled, it finds the best for the given data.
   - This approach, along with the native pandas integration across modules, ensures that BioNeuralNet can be easily incorporated into your analysis workflows.

**Metrics**:

- Several plotting funcctions to visualize networks, emebddings, variance distribution, cluster comparison, and more.
- Correlation based functions to compare clustersand omics data with the phenotype.

**Utilities**:
   
- **Filtering Functions**:

   - Network filtering allows users to select variance or zero-fraction filtering to an omics network.
   - Reducing noise, and removing outliers.

- **Data Conversion**:

   - Convert RData files both CSV and to Pandas DataFrame. For ease of integration for R-based workflows.

**External Tools**:

- **Graph Construction**:

   - BioNeuralNet provides additional tools in the [External Tools](https://bioneuralnet.readthedocs.io/en/latest/external_tools/index.html) module.
   - Allowing users to generate networks using R-based tools like WGCNA and SmCCNet.
   - While optional, these tools enhance BioNeuralNet's capabilities and are recommended for comprehensive analysis.

## 3. Quick Example: SmCCNet + DPMON for Disease Prediction

```python
import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.downstream_task import DPMON

# 1. Load dataset
loader = DatasetLoader("example1")
omics1, omics2, phenotype, clinical = loader.load_data()

# 2. Generate adjacency matrix using SmCCNet
smccnet = SmCCNet(
    phenotype_df=phenotype,
    omics_dfs=[omics1, omics2],
    data_types=["genes", "proteins"],
    kfold=3,
    subSampNum=500,
)
global_network, _ = smccnet.run()

# 3. Run Disease Prediction using DPMON
dpmon = DPMON(
    adjacency_matrix=global_network,
    omics_list=[omics1, omics2],
    phenotype_data=phenotype,
    clinical_data=clinical,
    tune=True,
)
dpmon_predictions = dpmon.run()
print("Disease Predictions:\n", dpmon_predictions.head())
```

## 4. Documentation and Tutorials

- Full documentation: [BioNeuralNet Documentation](https://bioneuralnet.readthedocs.io/en/latest/)
- Tutorials include:
  - Multi-omics graph construction
  - GNN embeddings for disease prediction
  - Subject representation with integrated embeddings
  - Clustering using Hybrid Louvain and Correlated PageRank
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

We also acknowledge R-based tools for external network construction:

- [**SmCCNet**](https://github.com/UCD-BDLab/BioNeuralNet/tree/main/bioneuralnet/external_tools/smccnet) - Sparse multiple canonical correlation network.
- [**WGCNA**](https://cran.r-project.org/web/packages/WGCNA/) - Weighted gene co-expression network analysis.

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

For more details, see our [Contributing Guide](https://github.com/UCD-BDLab/BioNeuralNet/blob/main/CONTRIBUTING.md).

## 9. License

BioNeuralNet is distributed under the [MIT License](https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE).

## 10. Contact

- **Issues and Feature Requests:** [Open an Issue](https://github.com/UCD-BDLab/BioNeuralNet/issues)
- **Email:** [vicente.ramos@ucdenver.edu](mailto:vicente.ramos@ucdenver.edu)
