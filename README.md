# BioNeuralNet: A Multi-Omics Integration and GNN-Based Embedding Framework

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyPI](https://img.shields.io/pypi/v/bioneuralnet)
![Python Versions](https://img.shields.io/pypi/pyversions/bioneuralnet)
![GitHub Issues](https://img.shields.io/github/issues/UCD-BDLab/BioNeuralNet)
![GitHub Contributors](https://img.shields.io/github/contributors/UCD-BDLab/BioNeuralNet)

## Welcome to BioNeuralNet Beta 0.1

![BioNeuralNet Logo](/assets/LOGO_WB.png)

**Note:** This is a **beta version** of BioNeuralNet. It is under active development, and certain features
may be incomplete or subject to change. Feedback and bug reports are highly encouraged to help us
improve the tool.

BioNeuralNet is a Python-based software tool designed to streamline the integration of multi-omics
data with **Graph Neural Network (GNN)** embeddings. It supports **graph clustering**, **subject representation**,
and **disease prediction**, enabling advanced analyses of complex multi-omics networks.

![BioNeuralNet Workflow](assets/BioNeuralNet.png)

---

## Key Features

BioNeuralNet offers five core steps in a typical workflow:

### 1. Graph Construction
- **Not** performed internally. You provide or build adjacency matrices externally (e.g., via **WGCNA**, **SmCCNet**, or your own scripts).
- Lightweight wrappers are available in `bioneuralnet.external_tools` (e.g., WGCNA, SmCCNet) for convenience. However, using these wrappers is **optional** and not mandatory for BioNeuralNet’s pipeline.

### 2. Graph Clustering
- Identify functional modules or communities using **PageRank**.
- The `PageRank` module enables finding subnetwork clusters through personalized sweep cuts, capturing local neighborhoods influenced by seed nodes.

### 3. Network Embedding
- Generate embeddings using methods like **GCN**, **GAT**, **GraphSAGE**, and **GIN**.
- You can attach numeric labels to nodes or remain “unsupervised,” relying solely on graph structure and node features (e.g., correlation with clinical data).

### 4. Subject Representation
- Integrate node embeddings back into omics data, enriching each subject’s feature vector by weighting columns with the learned embedding scalars.

### 5. Downstream Tasks
- Perform advanced analyses, such as disease prediction, via **DPMON**, which trains a GNN end-to-end alongside a classifier to incorporate both local and global network information.

---

# Installation

BioNeuralNet supports Python 3.10 and 3.11 in this beta release. Follow the steps below to set up BioNeuralNet and its dependencies.

## 1. Install BioNeuralNet via pip

To install the core BioNeuralNet modules for GNN embeddings, subject representation, disease prediction (DPMON), and clustering:

```bash
pip install bioneuralnet==0.1.0b1
```

## 2. Install PyTorch and PyTorch Geometric (Separately)

BioNeuralNet relies on PyTorch and PyTorch Geometric for GNN operations:

- Install PyTorch:
  ```bash
  pip install torch torchvision torchaudio
  ```
- Install PyTorch Geometric:
  ```bash
  pip install torch_geometric
  ```

For GPU-accelerated builds or other configurations, visit their official guides:
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Select the appropriate build for your system (e.g., Stable, Linux, pip, Python, CPU/GPU).

![PyTorch Installation](assets/pytorch.png)
*PyTorch Installation Guide*

![PyTorch Geometric Installation](assets/geometric.png)
*PyTorch Geometric Installation Guide*

## 3. (Optional) Install R and External Tools

If you plan to use **WGCNA** or **SmCCNet** for network construction:

- Install R from [The R Project](https://www.r-project.org/).
- Install the required R packages. Open R and run the following:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
install.packages(c("dplyr", "jsonlite"))
BiocManager::install(c("impute", "preprocessCore", "GO.db", "AnnotationDbi"))
install.packages("SmCCNet")
install.packages("WGCNA")
```

## 4. Additional Notes for External Tools

For **Node2Vec**, **feature selection**, or **visualization modules**, refer to the external tools documentation in `bioneuralnet.external_tools`. Examples include:
- **`Node2Vec`**: Node2Vec-based embeddings.
- **`FeatureSelector`**: Basic feature selection strategies like LassoCV and random forest.
- **`HierarchicalClustering`**: Agglomerative clustering and silhouette scoring.
- **`StaticVisualizer` and `DynamicVisualizer`**: Static or interactive network visualization.
- **`SmCCNet` / `WGCNA`**: Build adjacency matrices using R-based libraries.

These integrations are **optional** and do **not** form part of the core pipeline.

## 5. Development Setup (Optional)

If you plan to contribute to BioNeuralNet:

```bash
git clone https://github.com/UCD-BDLab/BioNeuralNet.git
cd BioNeuralNet
pip install -r requirements-dev.txt
pytest
```

---

## Quick Example: Transforming Multi-Omics for Enhanced Disease Prediction

![BioNeuralNet Overview](assets/Overview.png)
*BioNeuralNet: Transforming Multi-Omics for Enhanced Disease Prediction*

### External Tools

We offer a number of external tools available through the `bioneuralnet.external_tools` module:
- These tools were implemented to facilitate testing and should not be considered part of the package's core functionality.
- The classes inside the `external_tools` module are lightweight wrappers around existing tools and libraries offering minimal functionality.
- We highly encourage users to explore these tools outside of BioNeuralNet to fully leverage their capabilities.

### Steps

Below is a quick example demonstrating the following:

1. **Building or Importing a Network Adjacency Matrix**:
   - For instance, using external tools like **SmCCNet**.

2. **Using DPMON for Disease Prediction**:
   - A detailed explanation follows.

#### 1. Data Preparation
- Input your multi-omics data (e.g., proteomics, metabolomics, genomics) along with phenotype data.

#### 2. Network Construction
- **Not performed internally**: You need to provide or build adjacency matrices externally (e.g., via WGCNA, SmCCNet, or your own scripts).
- Lightweight wrappers are available in `bioneuralnet.external_tools` (e.g., WGCNA, SmCCNet) for convenience. However, using these wrappers is **optional** and not mandatory for BioNeuralNet’s pipeline.

#### 3. Disease Prediction
- **DPMON** integrates GNN-based node embeddings with a downstream neural network to predict disease phenotypes.

### Code Example:

```python
import pandas as pd
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.downstream_task import DPMON

# Step 1: Data Preparation
phenotype_data = pd.read_csv('phenotype_data.csv', index_col=0)
omics_proteins = pd.read_csv('omics_proteins.csv', index_col=0)
omics_metabolites = pd.read_csv('omics_metabolites.csv', index_col=0)
clinical_data = pd.read_csv('clinical_data.csv', index_col=0)

# Step 2: Network Construction
smccnet = SmCCNet(
    phenotype_df=phenotype_data,
    omics_dfs=[omics_proteins, omics_metabolites],
    data_types=["protein", "metabolite"],
    kfold=5,
    summarization="PCA",
)
adjacency_matrix = smccnet.run()
print("Adjacency matrix generated.")

# Step 3: Disease Prediction
dpmon = DPMON(
    adjacency_matrix=adjacency_matrix,
    omics_list=[omics_proteins, omics_metabolites],
    phenotype_data=phenotype_data,
    clinical_data=clinical_data,
    model="GAT",
)
predictions = dpmon.run()
print("Disease phenotype predictions:\n", predictions)
```

### Output
- **Adjacency Matrix**: The multi-omics network representation.
- **Predictions**: Disease phenotype predictions for each sample as a DataFrame linking subjects to predicted classes.

---

## Documentation & Tutorials

- Extensive documentation at [Read the Docs](https://bioneuralnet.readthedocs.io/en/latest/index.html)
- Tutorials illustrating:
  - Unsupervised vs. label-based GNN usage
  - PageRank clustering and hierarchical clustering
  - Subject representation
  - Integrating external tools like WGCNA, SmCCNet

## Frequently Asked Questions (FAQ)

Key topics include:
- **GPU acceleration** vs. CPU-only
- **External Tools** usage (R-based adjacency construction, Node2Vec, etc.)
- **DPMON** for local/global structure-based disease prediction
- **PageRank or HierarchicalClustering** for subnetwork identification

See [FAQ](https://bioneuralnet.readthedocs.io/en/latest/faq.html) for more.

---

## Acknowledgments

BioNeuralNet relies on or interfaces with various open-source libraries:

- [PyTorch](https://pytorch.org/) / [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [Node2Vec](https://github.com/aditya-grover/node2vec)
- [WGCNA](https://cran.r-project.org/package=WGCNA) / [SmCCNet](https://cran.r-project.org/package=SmCCNet)
- [Pytest](https://pytest.org/), [Sphinx](https://www.sphinx-doc.org), [Black](https://black.readthedocs.io/), [Flake8](https://flake8.pycqa.org/)

We appreciate the efforts of these communities and all contributors.

## Testing & CI

1. **Local Testing**:
   ```bash
   pytest --cov=bioneuralnet --cov-report=html
   open htmlcov/index.html
   ```

2. **Continuous Integration**:
   - GitHub Actions run our test suite and code checks on each commit and PR.

## Contributing

- **Fork** the repository, create a new branch, implement your changes.
- **Add/Update** tests, docstrings, and examples if appropriate.
- **Open** a pull request describing your modifications.

For more details, see our [FAQ](https://bioneuralnet.readthedocs.io/en/latest/faq.html)
or open an [issue](https://github.com/UCD-BDLab/BioNeuralNet/issues).

## License & Contact

- **License**: [MIT License](https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE)
- **Contact**: Questions or feature requests? [Open an issue](https://github.com/UCD-BDLab/BioNeuralNet) or email [vicente.ramos@ucdenver.edu](mailto:vicente.ramos@ucdenver.edu).

---

BioNeuralNet aims to **streamline** multi-omics network analysis by providing **graph clustering**, **GNN embedding**, **subject representation**, and **disease prediction** tools. We hope it helps uncover new insights in multi-omics research.
