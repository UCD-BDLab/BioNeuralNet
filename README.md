# BioNeuralNet: A Multi-Omics Integration and GNN-Based Embedding Framework

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyPI](https://img.shields.io/pypi/v/bioneuralnet)
![GitHub Issues](https://img.shields.io/github/issues/UCD-BDLab/BioNeuralNet)
![GitHub Contributors](https://img.shields.io/github/contributors/UCD-BDLab/BioNeuralNet)
![Downloads](https://static.pepy.tech/badge/bioneuralnet)

[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue.svg)](https://bioneuralnet.readthedocs.io/en/latest/)

## Welcome to [BioNeuralNet Beta 0.2](https://bioneuralnet.readthedocs.io/en/latest/index.html)

![BioNeuralNet Logo](/assets/LOGO_WB.png)

**Note:** This is a **beta version** of BioNeuralNet. We are actively developing new features and improving stability.  
Feedback and bug reports are highly encouraged!

BioNeuralNet is a Python framework for integrating **multi-omics data** with **Graph Neural Networks (GNNs)**.  
It provides tools for **graph construction, clustering, network embedding, subject representation, and disease prediction**.

![BioNeuralNet Workflow](assets/BioNeuralNet.png)

---

## **Key Features**

BioNeuralNet enables **multi-omics analysis** through **five core steps**:

### 1. **Graph Construction**
- Build multi-omics networks using **SmCCNet** or custom adjacency matrices.

### 2. **Graph Clustering**
- Identify meaningful communities with **Correlated Louvain**, **Hybrid Louvain**, or **Correlated PageRank**.

### 3. **GNN Embedding**
- Generate node embeddings using models like **GCN, GAT, GraphSAGE, and GIN**.

### 4. **Subject Representation**
- Integrate GNN-based embeddings into omics data via **GraphEmbedding**.

### 5. **Disease Prediction**
- Use **DPMON**, a GNN-powered classifier, to predict disease phenotypes.

---

## **Installation**

BioNeuralNet supports **Python 3.10 and 3.11**.  

### **1. Install BioNeuralNet**
```bash
pip install bioneuralnet
```

### **2. Install PyTorch and PyTorch Geometric**
BioNeuralNet relies on PyTorch for GNN computations. Install PyTorch **separately**:

- **PyTorch (CPU)**:
  ```bash
  pip install torch torchvision torchaudio
  ```

- **PyTorch Geometric**:
  ```bash
  pip install torch_geometric
  ```

For GPU acceleration, visit:
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

---

## **Quick Example: SmCCNet + DPMON for Disease Prediction**

```python
import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.downstream_task import DPMON
import pandas as pd

# 1) Load dataset
loader = DatasetLoader("example1")
omics1, omics2, phenotype, clinical = loader.load_data()

# 2) Generate adjacency matrix using SmCCNet
smccnet = SmCCNet(
    phenotype_df=phenotype,
    omics_dfs=[omics1, omics2],
    data_types=["genes", "proteins"],
    kfold=3,
    subSampNum=500,
)
global_network, _ = smccnet.run()

# 3) Run Disease Prediction using DPMON
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

### **Output**
- **Adjacency Matrix**: The constructed multi-omics network.
- **Predictions**: Disease phenotype predictions.

---

## **Documentation & Tutorials**
- Full documentation: [Read the Docs](https://bioneuralnet.readthedocs.io/en/latest/)
- Tutorials include:
  - Multi-omics graph construction
  - GNN embeddings for disease prediction
  - Subject representation with integrated embeddings
  - Clustering using Hybrid Louvain & Correlated PageRank

---

## **Frequently Asked Questions (FAQ)**

- Does BioNeuralNet support **GPU acceleration**?  
   - Yes, install PyTorch with CUDA support.

- Can I use my own **adjacency matrix**?  
 - yes, you can provide a custom matrix instead of using SmCCNet.

- What clustering methods are supported?  
   - **Correlated Louvain**, **Hybrid Louvain**, and **Correlated PageRank**.

See the full [FAQ](https://bioneuralnet.readthedocs.io/en/latest/faq.html).

---

## **Acknowledgments**

BioNeuralNet integrates multiple open-source libraries. We acknowledge key dependencies:

- **PyTorch** - GNN computations and deep learning models.  
- **PyTorch Geometric** - Graph-based learning for multi-omics.  
- **NetworkX** - Graph data structure and algorithms.  
- **Scikit-learn** - Feature selection and evaluation utilities.  
- **pandas & numpy** - Core data processing tools.  
- **ray[tune]** - Hyperparameter tuning for GNN models.  
- **matplotlib** - Data visualization.  
- **cptac** - Dataset handling for clinical proteomics.  
- **python-louvain** - Community detection algorithms.  

We also acknowledge R-based tools for external network construction:
- **SmCCNet** - Sparse multiple canonical correlation network.
- **WGCNA** - Weighted gene co-expression network analysis.

These tools **enhance BioNeuralNet** but are **not required** for core functionality.

---

## **Testing & Continuous Integration**

1. **Run Tests Locally**:
   ```bash
   pytest --cov=bioneuralnet --cov-report=html
   open htmlcov/index.html
   ```

2. **Continuous Integration**:
   - GitHub Actions runs automated tests on each commit.

---

## **Contributing**

We welcome contributions!  
To get started:

```bash
git clone https://github.com/UCD-BDLab/BioNeuralNet.git
cd BioNeuralNet
pip install -r requirements-dev.txt
pre-commit install
pytest
```

### **How to Contribute**
- **Fork** the repo, create a new branch, implement your changes.
- **Add tests and documentation** for your new feature.
- **Submit a pull request** with a clear description.

For more details, check our [Contributing Guide](https://github.com/UCD-BDLab/BioNeuralNet/blob/main/CONTRIBUTING.md).

---

## **License**
- **License:** [MIT License](https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE)

---

## **Contact**
- **Issues & Feature Requests:** [Open an Issue](https://github.com/UCD-BDLab/BioNeuralNet/issues)
- **Email:** [vicente.ramos@ucdenver.edu](mailto:vicente.ramos@ucdenver.edu)

