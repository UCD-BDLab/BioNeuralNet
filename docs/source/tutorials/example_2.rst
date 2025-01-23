Example 2: SmCCNet + DPMON for Disease Prediction
=================================================

This tutorial illustrates how to:

1. **Build** an adjacency matrix with SmCCNet (external).
2. **Predict** disease phenotypes using DPMON.

**Workflow**:

1. **Data Preparation**:
   - Multi-omics data, phenotype data with disease labels, and (optionally) clinical data.

2. **Network Construction**:
   - Use `SmCCNet` to create an adjacency matrix from the combined omics data.

3. **Disease Prediction**:
   - `DPMON` integrates the adjacency matrix, omics data, and phenotype to train a GNN + classifier end-to-end.

4. **Diagram of the workflow**: The figure below illustrates the process.

`View full-size image: Disease Prediction (DPMON) <https://bioneuralnet.readthedocs.io/en/latest/_images/DPMON.png>`_

.. figure:: _static/DPMON.png
   :align: center
   :alt: Disease Prediction (DPMON)

   Embedding-enhanced subject data using DPMON for improved disease prediction.

.. code-block:: python

   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON
   import pandas as pd

   # 1) Prepare data
   omics_data = pd.read_csv("data/omics_data.csv")
   phenotype_data = pd.read_csv("data/phenotype_data.csv")
   clinical_data = pd.read_csv("data/clinical_data.csv")

   # 2) Run SmCCNet to get adjacency
   smccnet = SmCCNet(
      phenotype_df=phenotype_data,
      omics_df=omics_data,
      data_types=["genes, proteins"]
      kfolds=5,
      summarization = "NetSHy",
      seed: 127,
      )
   adjacency_matrix = smccnet.run()

   # 3) Disease Prediction with DPMON
   dpmon = DPMON(
      adjacency_matrix=adjacency_matrix,
      omics_list=[omics_data],
      phenotype_data=phenotype_data,
      clinical_data=clinical_data,
      model: "GAT",
      gnn_hidden_dim: 64,
      layer_num: 3,
      nn_hidden_dim1: 2,
      nn_hidden_dim2: 2,
      epoch_num: 10,
      repeat_num: 5,
      lr: 0.01,
      weight_decay: 1e-4,
      tune: True,
      gpu: False
   )
   predictions = dpmon.run()
   print("Disease predictions:\n", predictions)

**Output**:
- **Adjacency Matrix**: from SmCCNet
- **Predictions**: Phenotype predictions for each subject
