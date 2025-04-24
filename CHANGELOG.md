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

## [1.0.1] to [1.0.4] - 2025-04-24

- **BUG**: A bug related to rdata files missing
- **New realease**: A new release will include documentation for the other updates. (1.1.0)