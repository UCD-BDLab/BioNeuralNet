Example 2: SmCCNet + DPMON for Disease Prediction
=================================================

This tutorial illustrates how to:

1. **Build** an adjacency matrix with SmCCNet.
2. **Predict** disease phenotypes using DPMON.

**Workflow**:

1. **Data Preparation**:
   - Load multi-omics, phenotype, and clinical data using DatasetLoader.

2. **Network Construction**:
   - Use `SmCCNet.run()` to create an adjacency matrix from the combined omics data.

3. **Disease Prediction**:
   - `DPMON` integrates the adjacency matrix, omics data, and phenotype data to train a GNN-based classifier.

4. **Diagram of the workflow**: The figure below illustrates the process.

.. figure:: ../_static/DPMON.png
   :align: center
   :alt: Disease Prediction (DPMON)

   Embedding-enhanced subject data using DPMON for improved disease prediction.

`View full-size image: Disease Prediction (DPMON) <https://bioneuralnet.readthedocs.io/en/latest/_images/DPMON.png>`_

**Step-by-Step Instructions**:

1. **Data Setup**:
   - Load synthetic multi-omics, phenotype, and clinical data using `DatasetLoader`.

2. **Network Construction (SmCCNet)**:
   - Call `SmCCNet.run()` to produce an adjacency matrix from the omics data.

3. **Disease Prediction (DPMON)**:
   - Pass the adjacency, omics, phenotype, and clinical data into `DPMON`.
   - Run `.run()` to predict disease phenotypes.

Below is a **complete** Python implementation:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.datasets import DatasetLoader
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON

   # 1) Load dataset
   loader = DatasetLoader("example1")
   omics1, omics2, phenotype, clinical = loader.load_data()

   # 2) Merge omics data
   merged_omics = pd.concat([omics1, omics2], axis=1)

   # 3) Generate adjacency matrix using SmCCNet
   smccnet = SmCCNet(
       phenotype_df=phenotype,
       omics_dfs=[omics1, omics2],
       data_types=["genes", "proteins"],
       kfold=3,
       subSampNum=500,
   )
   global_network, smccnet_clusters = smccnet.run()

   print("Adjacency matrix generated using SmCCNet.")

   # 4) Run Disease Prediction using DPMON
   dpmon = DPMON(
       adjacency_matrix=global_network,
       omics_list=[omics1, omics2],
       phenotype_data=phenotype,
       clinical_data=clinical,
       tune=True,
   )
   dpmon_predictions = dpmon.run()

   print("\nDPMON Predictions:")
   print(dpmon_predictions.head())

**Output**:
- **Adjacency Matrix**: Generated using SmCCNet.
- **Predictions**: Phenotype predictions for each subject.
