Notebooks
=========

This collection of demonstration notebooks provides a reproducible benchmarkfor multi-omics classification and subgraph detection using the **BioNeuralNet framework**.

Each workflow walks through:

* **Feature selection** and data preprocessing.
* **Network construction** (similarity graphs and phenotype-aware networks).
* **Hyperparameter optimization** for Graph Neural Networks (GNNs).
* **Downstream tasks**, including disease prediction with DPMON.
* **Subgraph detection and biomarker modules** in selected cohorts.

The TCGA notebooks showcase the full end-to-end pipeline on multiple cancers (BRCA, LGG, KIPAN, PAAD). The biomarker notebook focuses on phenotype-associated subgraphs and driver modules derived from these analyses.

The following notebooks are included in this guide:

.. toctree::
   :maxdepth: 1

   TCGA-BRCA.ipynb
   TCGA-LGG.ipynb
   TCGA-KIPAN.ipynb
   TCGA-PAAD.ipynb
   TCGA-Biomarkers.ipynb
