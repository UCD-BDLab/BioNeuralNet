Notebooks
=========

This collection of demonstration notebooks provides a reproducible benchmarkfor multi-omics classification and subgraph detection using the **BioNeuralNet framework**.

Each workflow walks through:

* **Feature selection** and data preprocessing.
* **Network construction** (similarity graphs and phenotype-aware networks).
* **Hyperparameter optimization** for Graph Neural Networks (GNNs).
* **Downstream tasks**, including disease prediction with DPMON.
* **Subgraph detection and biomarker modules** in selected cohorts.

The TCGA notebooks showcase the full end-to-end pipeline on multiple cancers (BRCA, LGG, KIPAN). The TCGA-LGG notebook includes a biomarker discovery section through phenotype-associated subgraphs and driver modules.

The following notebooks are included in this guide:

.. toctree::
   :maxdepth: 1

   TCGA-BRCA.ipynb
   TCGA-LGG.ipynb
   TCGA-KIPAN.ipynb
   ROSMAP.ipynb
