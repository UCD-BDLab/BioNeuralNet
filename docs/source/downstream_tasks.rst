Downstream Tasks
================

BioNeuralNet leverages **graph neural network (GNN)-based embeddings** to transform complex multi-omics networks into compact, biologically meaningful representations. These embeddings serve as a powerful foundation for diverse downstream analyses, such as disease prediction, enhanced subject-level profiling, biomarker discovery, and exploratory visualization.

Core Downstream Applications
----------------------------

By capturing both structural and functional relationships inherent in multi-omics data, BioNeuralNet-generated embeddings unlock key applications:

- **Disease Prediction**: Leverage network-derived embeddings in an end-to-end pipeline (DPMON) for accurate disease classification.
- **Enhanced Subject Representation**: Integrate embeddings with original omics data to improve predictive modeling and patient stratification.
- **Exploratory Analysis**: Facilitate visualization, biomarker identification, and phenotype-associated clustering using low-dimensional embeddings.

.. image:: _static/Overview.png
   :align: center
   :alt: BioNeuralNet Embedding Applications
   :width: 100%

Disease Prediction (DPMON)
--------------------------

BioNeuralNet **DPMON module** provides a streamlined, end-to-end framework for disease classification tasks. It integrates multi-omics data, phenotype-informed network embeddings, and clinical covariates to deliver robust predictive models with optional automated hyperparameter tuning.

.. image:: _static/DPMON.png
   :align: center
   :alt: DPMON Disease Prediction Workflow
   :width: 100%

**Example Usage**:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON
   from bioneuralnet.datasets import DatasetLoader

   # Step 1: Load sample data
   example = DatasetLoader("example1")
   omics_genes = example.data["X1"]
   omics_proteins = example.data["X2"]
   phenotype = example.data["Y"]
   clinical = example.data["clinical_data"]

   # Step 2: Construct phenotype-aware network
   smccnet = SmCCNet(
       phenotype_df=phenotype,
       omics_dfs=[omics_genes, omics_proteins],
       data_types=["Genes", "Proteins"],
       kfold=5,
       summarization="PCA",
   )
   global_network, clusters = smccnet.run()

   # Step 3: Disease prediction with DPMON
   dpmon = DPMON(
       adjacency_matrix=global_network,
       omics_list=[omics_genes, omics_proteins],
       phenotype_data=phenotype,
       clinical_data=clinical,
       model="GCN",
   )
   predictions, avg_accuracy = dpmon.run()
   print("Disease phenotype predictions:\n", predictions)

Enhanced Subject Representation
-------------------------------

Beyond predictive modeling, BioNeuralNet embeddings can be integrated directly into subject-level multi-omics data to enhance the discriminative power and interpretability of patient profiles. This enriched representation supports several analytical tasks:

- **Biomarker Discovery**: Highlight key molecular features strongly associated with clinical outcomes.
- **Patient Stratification**: Improve clustering and subgroup identification based on embedding-enhanced profiles.
- **Visualization and Interpretation**: Facilitate intuitive exploration of high-dimensional data through embedding-based visualizations.

.. image:: _static/SubjectRepresentation.png
   :align: center
   :alt: Subject-Level Embedding Integration
   :width: 100%

Embedding-Based Exploratory Analysis
------------------------------------

BioNeuralNet low-dimensional embeddings simplify complex omics relationships, providing intuitive entry points into exploratory data analysis, including:

- **Community Detection**: Uncover biologically relevant clusters linked to clinical phenotypes.
- **Feature Importance Analysis**: Evaluate embedding contributions to predictive models, enhancing interpretability.
- **Interactive Visualization**: Integrate embeddings seamlessly with common Python visualization libraries (e.g., Matplotlib, Seaborn) for insightful plots and network representations.

Customization and Extensibility
-------------------------------

BioNeuralNet is designed with modularity and flexibility in mind. Users can easily adapt embedding outputs for custom analytical workflows, integrating them into broader bioinformatics pipelines or developing novel downstream applications tailored to specific research goals.

Getting Started
---------------

To explore comprehensive end-to-end analyses and practical tutorials, refer to:

- :doc:`Quick_Start`
- :doc:`notebooks/index`

References
----------
Further methodological details and model insights can be found in our documentation and accompanying publications.

Return to :doc:`../index`
