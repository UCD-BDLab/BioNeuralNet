# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0b2] - 2025-02-16

### **Added**
- **Hybrid Louvain Clustering**: New iterative Louvain clustering method with PageRank refinement.
- **DatasetLoader Integration**: Standardized dataset loading for examples & tutorials.
- **GNNEmbedding Improvements**: Enhanced tuning capabilities for better model selection.
- **New Example Workflows**:
  - **SmCCNet + GNN Embeddings + Subject Representation**
  - **SmCCNet + DPMON for Disease Prediction**
  - **Hybrid Louvain Clustering Example**
- **Updated Tutorials**: Expanded documentation with `nbsphinx` support for `.ipynb` notebooks.

### **Changed**
- **FAQ Overhaul**: Simplified and updated FAQs based on new features.
- **Improved Documentation**: Rewrote and updated multiple sections for better clarity.
- **Updated README.md**:
  - Improved feature descriptions.
  - New **quick-start example** for SmCCNet + DPMON.
  - Cleaner installation instructions.
- **Refactored Examples**:
  - Updated **clustering** and **embedding** workflows to match the latest API changes.

### **Fixed**
- **Bug Fixes**:
  - Resolved incorrect handling of `tune=True` in Hybrid Louvain.
  - Addressed inconsistencies in `GraphEmbedding` parameter parsing.
  - Fixed dataset loading issues in example scripts.

## **[Unreleased]**
- Multi-Modal Integration

## [1.0] - 2025-03-06

### **Added**
- **Simplified requirements**: Requirements.txt was severely simplified. Addtionally removed unnecessary imports from core package
- **New Metrics**: New correlation, evaluations and plot python files
  - **Plotting Functions**:
    - plot_variance_distribution
    - plot_variance_by_feature
    - plot_performance
    - plot_embeddings
    - plot_network
    - compare_clusters
  - **Correlation Functions**
    - omics_correlation
    - cluster_correlation
    - louvain_to_adjacency
  - **Evaluation**
    - evaluate_rf
- **New Utilities**: Added files to convert RData (Networks as adjency matrix) files to Pandas Dataframes Adjancy matrix.
  - **Variance Functions**:
    - remove_variance
    - remove_fraction
    - network_remove_low_variance
    - network_remove_high_zero_fraction
    - network_filter
    - omics_data_filter

- **Updated Tutorials and Documentation**: New end to end jupiter notebook example.
- **Updated Test**: All test have been updated and new ones have been added.

## [1.1.1] - 2025-07-12

### **Added**
- **New Embedding Integration Utility**
  - `_integrate_embeddings(reduced, method="multiply", alpha=2.0, beta=0.5)`:
    - Integrates reduced embeddings with raw omics features via a multiplicative scheme:
    - `enhanced = beta * raw + (1 - beta) * (alpha * normalized_weight * raw)`
    - (default ensures ≥ 50 % of each feature’s final value is influenced by the learned weights).

- **Graph-Generation Algorithms**
  - `gen_similarity_graph`: k-NN Cosine / Gaussian RBF similarity graph
  - `gen_correlation_graph`: Pearson / Spearman co-expression graph
  - `gen_threshold_graph`: soft-threshold (WGCNA-style) correlation graph
  - `gen_gaussian_knn_graph`: Gaussian kernel k-NN graph
  - `gen_mutual_info_graph`: mutual-information graph

- **Preprocessing Utilities**
  - Clinical data pipeline `preprocess_clinical`
  - Inf/NaN cleaning: `clean_inf_nan`
  - Variance selection: `select_top_k_variance`
  - Correlation selection (supervised / unsupervised): `select_top_k_correlation`
  - RandomForest importance: `select_top_randomforest`
  - ANOVA F-test selection: `top_anova_f_features`
  - Network-pruning helpers:
      - `prune_network`, `prune_network_by_quantile`,
      - `network_remove_low_variance`, `network_remove_high_zero_fraction`

- **Continuous-Deployment Workflow**
  Added `.github/workflows/publish.yml` to auto-publish releases to PyPI when a Git tag is pushed.

- **Updated Homepage Image**
  Replaced the index-page illustration to depict the full BioNeuralNet workflow.

### **Changed**
- **Comprehensive Documentation Update**
  - Rebuilt ReadTheDocs site with a new workflow diagram on the landing page.
  - Synced API reference to include all new graph-generation, preprocessing, and embedding-integration functions.
  - Added quick-start guide, expanded tutorials, and refreshed examples/notebooks.
  - Updated narrative docs, docstrings, and licencing info for consistency.

- **License**: Project is now distributed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### **Fixed**
- **Packaging Bug**: Missing `.csv` datasets and `.R` scripts in source distribution; `MANIFEST.in` updated to include all requisite data files.

## [1.1.2] - 2025-11-01

- **Linked Zenodo DOI to GitHub repository**

## [1.1.3] - 2025-11-01

- **Tag update to Sync Zenodo and PIPY**

## [1.1.4] - 2025-11-08

### **Added**
- **New Utility Functions and Global Seed**

  - **Imputation**:
    - `impute_omics`: Imputes missing values (NaNs) using `mean`, `median`, or `zero` strategies.

    - `impute_omics_knn`: Implements **K-Nearest Neighbors (KNN)** imputation for missing values in omics data.

  - **Normalization/Scaling**:
    - `normalize_omics`: Scales feature data using `Z-score`, `MinMax` scaling, or `Log2` transformation.

  - **Methylation Data Transformation**:
      - `beta_to_m`: Converts methylation `Beta-values to M-values` (log2 transformation).

  - **Reproducibility Utility**:
    - `set_seed`: Sets global random seeds across Python, NumPy, and PyTorch, including configuration for deterministic CUDA operations to ensure maximum experimental reproducibility.

- **New TCGA Datasets**

  - **TCGA-KIPAN**: Added the Pan-Kidney cohort (KIRC, KIRP, KICH) to **DatasetLoader**.

  - **TCGA-GBMLGG**: Added the combined Glioblastoma Multiforme and Lower-Grade Glioma cohort to **DatasetLoader**.

### **Changed**

  - **Documentation Update**: Updated the online documentation (Read the Docs/API Reference) to include the new TCGA datasets and their respective classification results using the **DPMON**.

## [1.2.0] - 2025-11-23

### API and Architecture Refactoring
- **Namespace Hierarchy Overhaul**: Transitioned from a flat namespace to a hybrid hierarchical structure to enhance modularity and prevent namespace pollution.
    - **Core Classes**: `DPMON`, `GNNEmbedding`, `SubjectRepresentation`, `SmCCNet`, and `DatasetLoader` remain accessible at the top level (e.g., `bnn.DPMON`).
    - **Utilities and Metrics**: Functional tools are now scoped to their respective submodules (e.g., `bnn.metrics.plot_network`, `bnn.utils.preprocess_clinical`).
- **Utils Module Restructuring**: Decomposed the monolithic `utils` module into specialized submodules for improved maintainability:
    - `utils.data`: Contains summary statistics functions (e.g., `variance_summary`).
    - `utils.preprocess`: Contains data transformation functions (e.g., `impute_omics`, `normalize_omics`).
    - `utils.reproducibility`: Dedicated module for seeding functions (`set_seed`).

### New Features
- **Graph Engineering Module (`graph_tools`)**: Introduced a new module for the diagnosis and repair of network topology issues.
    - `repair_graph_connectivity`: Implemented an algorithm to reconnect fragmented network components (islands) to the global network using eigenvector centrality hubs or omics-driven correlation.
    - `find_optimal_graph`: Added an AutoML-style search function that benchmarks various graph construction strategies (Gaussian, Correlation, Threshold) using a structural proxy task to optimize downstream stability.
    - `graph_analysis`: Added diagnostic utilities to log topological metrics (clustering coefficient, average degree) and identify isolated subgraphs broken down by omics modality.
- **DPMON Enhancements**: Expanded the `NeuralNetwork` backbone to support multiple dimensionality reduction strategies beyond the standard AutoEncoder.
    - **Linear Projection**: Added `ScalarProjection`, utilizing a linear layer to map embeddings to feature weights.
    - **MLP Projection**: Added `MLPProjection`, utilizing a non-linear Multilayer Perceptron for complex feature weighting.
- **Dataset Loaders**:
    - Implemented functional loaders (`load_brca`, `load_kipan`, `load_lgg`, `load_paad`, `load_monet`, `load_example`) to provide immediate access to data dictionaries, aligning with `scikit-learn` conventions.
    - Added `__getitem__` support to the `DatasetLoader` class for direct key access (e.g., `loader['rna']`).

### Data Standardization
- **BRCA Clinical Update**: Removed 15 duplicated columns from the BRCA clinical dataset, reducing the feature dimensionality from 118 to 103 to ensure data uniqueness.
- **Dataset Renaming**:
    - Renamed the synthetic dataset `example1` to `example`.
    - Renamed `gbmlgg` to `lgg` (Brain Lower Grade Glioma).
- **Target Variable Update**: Updated the target variable for the `lgg` dataset from 'histological type' to 'vital_status' to better align with prognostic prediction tasks.
- **Key Standardization**: Removed redundant `_data` suffixes from dataset dictionary keys (e.g., `monet['mirna_data']` is now `monet['mirna']`).
- **Dataset Specifications**: Updated documentation to explicitly define the dimensions (samples × features) for all included datasets:
    - **BRCA**: miRNA (769, 503), Target (769, 1), Clinical (769, 103), RNA (769, 2500), Meth (769, 2203).
    - **LGG**: miRNA (511, 548), Target (511, 1), Clinical (511, 13), RNA (511, 2127), Meth (511, 1823).
    - **PAAD**: CNV (177, 1035), Target (177, 1), Clinical (177, 19), RNA (177, 1910), Meth (177, 1152).
    - **KIPAN**: miRNA (658, 472), Target (658, 1), Clinical (658, 19), RNA (658, 2284), Meth (658, 2102).
    - **Monet**: Gene (107, 5039), miRNA (107, 789), Phenotype (106, 1), RPPA (107, 175), Clinical (107, 5).
    - **Example**: X1 (358, 500), X2 (358, 100), Y (358, 1), Clinical (358, 6).

### Improvements and Fixes
- **Documentation**: Refactored all docstrings across the library to adhere to strict Google Style formatting (Args/Returns) to ensure consistent API documentation generation.
- **Clustering**:
    - **Hybrid Louvain**: Corrected the parameter tuning logic for `k3` and `k4` weights and refined the iterative refinement loop for identifying phenotype-associated subgraphs.
    - **Correlated PageRank**: Enhanced input validation to ensure proper alignment between graph nodes and omics features.

### Removed
- **Metrics Evaluation**: Removed the `metrics.evaluation` module. Its functionality has been consolidated into the `metrics` module or deprecated in favor of external validation workflows.

### Left to Do
- **(DONE)Test Suite Completion**: Refactor remaining tests (`gnn_embedding`, `subject_representation`, `hybrid_louvain`) to align with new `utils` imports and other major changes.
- **Documentation**: Update ReadTheDocs API reference and `README.md` to reflect the split `utils` submodules and new `graph_tools`. All jupyter notebook examples will need to be updates.
- **(DONE)Release Prep**: Bump version to `1.2.0` in `setup.py`.
- **(DONE)Errors**: Errors with tests and doc-build are expected. They will be addresed in following smaller versions `1.2.1` and so on.
