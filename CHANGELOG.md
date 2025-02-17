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
