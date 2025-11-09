Notebooks
=========

This collection of demonstration notebooks establishes a robust and reproducible benchmark for multi-omics classification using the **BioNeuralNet framework**.

These workflows provide a comprehensive, step-by-step guide to:

* **Feature Selection** and data preprocessing.
* **Hyperparameter Optimization** for Graph Neural Networks (GNNs).
* **Downstream Tasks** including disease prediction with DPMON.

This documentation also includes dedicated guides for the framework's supporting utilities:

* **Dataset Loading**: Access the 5 included example datasets (TCGA-BRCA, TCGA-GBMLGG, TCGA-KIPAN, etc.).
* **Network Construction**: Demonstrates techniques for building omics networks.

.. note::
   The **TCGA-BRCA** notebook reflects an older version of the framework. The **TCGA-GBMLGG** and **TCGA-KIPAN** notebooks demonstrate the latest, end-to-end workflow.

The following notebooks are included in this guide:

.. toctree::
   :maxdepth: 1

   TCGA-BRCA.ipynb
   TCGA-GBMLGG.ipynb
   TCGA-KIPAN.ipynb
   datasets.ipynb
   network_construction.ipynb