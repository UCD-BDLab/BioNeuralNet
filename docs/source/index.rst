.. BioNeuralNet documentation master file

Welcome to BioNeuralNet Beta 0.1
================================

**Note:** This is a **beta version** of BioNeuralNet. It is under active development, and certain features 
may be incomplete or subject to change. Feedback and bug reports are highly encouraged to help us 
improve the tool.

BioNeuralNet is a Python-based software tool designed to streamline the transformation of multi-omics 
data into network-based representations and lower-dimensional embeddings, enabling advanced analytical 
processes like clustering, feature selection, disease prediction, and environmental exposure assessment.

`pip install bioneuralnet <https://pypi.org/project/bioneuralnet/>`__ to start using it. For details on installing, go to `installation.rst`_.

---

.. figure:: _static/Overview.png
   :align: center
   :alt: BioNeuralNet Overview
   :figwidth: 80%

   **BioNeuralNet**: Transforming Multi-Omics for Enhanced Disease Prediction

---

**Example Workflow: Multi-Omics Network to Disease Prediction** 
---------------------------------------------------------------

BioNeuralNet enables seamless integration of multi-omics data into a network-based analysis pipeline. 
Here’s a quick example demonstrating how to generate a network representation using SmCCNet and apply it 
to disease prediction using DPMON:

### Steps:

1. **Data Preparation**:
   - Input your multi-omics data (e.g., proteomics, metabolomics, genomics) along with phenotype data.

2. **Network Construction**:
   - Use Sparse Multiple Canonical Correlation Network (SmCCNet) to generate a network from the omics data. 
   - This step constructs an adjacency matrix capturing correlations and interactions between features.

3. **Disease Prediction**:
   - Disease Prediction using Multi-Omics Networks (DPMON) uses Graph Neural Networks (GNNs) to predict disease phenotypes.
   - Integrates multi-omics data and network structure information to generate GNNs embeddings that capture global and local graph patterns.
   - It enhances the Omics-data by creating enriched with node features. These are processed through a Neural Network, optimized end-to-end, enhancing predictive accuracy and reducing overfitting.

**Code Example**:

```python
   import pandas as pd
   from bioneuralnet.graph_generation import SmCCNet
   from bioneuralnet.downstream_task import DPMON

   # Step 1: Load Multi-Omics Dataset
   omics_data = pd.read_csv('omics_data.csv', index_col=0)
   phenotype_data = pd.read_csv('phenotype_data.csv', index_col=0)

   # Step 2: Generate a network using SmCCNet
   smccnet = SmCCNet(phenotype_data=phenotype_data, omics_data=omics_data)
   adjacency_matrix = smccnet.run()
   print("Multi-Omics Network generated.")

   # Step 3: Enhanced disease prediction using network information with DPMON
   dpmon = DPMON(adjacency_matrix=adjacency_matrix, omics_list=[omics_data], phenotype_data=phenotype_data)
   predictions = dpmon.run()
   print("Disease phenotype predictions:")
   print(predictions)
```


**Output**:
- **Adjacency Matrix**: The network representation of the multi-omics data.
- **Predictions**: Disease phenotype predictions for each sample.

---

For advanced subject representations using embeddings:

.. figure:: _static/SubjectRepresentation.png
   :align: center
   :alt: Subject Representation Workflow
   :figwidth: 80%

   Subject-level embeddings provide richer phenotypic and clinical context.

And for disease prediction tasks leveraging embeddings:

.. figure:: _static/DPMON.png
   :align: center
   :alt: Disease Prediction (DPMON)
   :figwidth: 80%

   Embedding-enhanced subject data support models like DPMON for improved disease prediction.

---

Documentation Overview
-----------------------

.. toctree::
   :maxdepth: 2

   installation
   tools/index
   tutorials/index
   api_reference
   faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
