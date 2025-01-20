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

.. code-block:: python

   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON
   import pandas as pd

   # 1) Load data
   omics_data = pd.read_csv("omics_data.csv", index_col=0)
   phenotype_data = pd.read_csv("phenotype_data.csv", index_col=0)

   # 2) Generate adjacency
   smcc = SmCCNet(phenotype_data=phenotype_data, omics_data=omics_data)
   adjacency_matrix = smcc.run()

   # 3) Disease Prediction with DPMON
   dpmon = DPMON(
       adjacency_matrix=adjacency_matrix,
       omics_list=[omics_data],
       phenotype_data=phenotype_data,
       model='GAT'
   )
   predictions = dpmon.run()
   print("Disease predictions:\n", predictions)

**Output**:
- **Adjacency Matrix**: from SmCCNet
- **Predictions**: Phenotype predictions for each subject
