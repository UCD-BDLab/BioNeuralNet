Tutorials
=========

BioNeuralNet offers comprehensive tutorials to help you understand and utilize its features effectively. Below are step-by-step guides to various use cases.

.. toctree::
   :maxdepth: 2
   :caption: Available Tutorials:

   tutorial_1_getting_started
   tutorial_2_embedding_generation
   tutorial_3_subject_representation
```

**Explanation:**

- **`toctree`**: Links to individual tutorial `.rst` files (e.g., `tutorial_1_getting_started.rst`). You need to create these files with detailed step-by-step instructions.

**Example: `docs/source/tutorial_1_getting_started.rst`**

```rst
Getting Started with BioNeuralNet
=================================

This tutorial guides you through the basic setup and initial usage of BioNeuralNet.

Step 1: Setup
-------------

Ensure that you have followed the [Installation](installation) guide to set up the environment.

Step 2: Importing the Package
-----------------------------

```python
import bioneuralnet as bn
```

Step 3: Loading Data
--------------------

Load your omics, phenotype, and clinical data.

```python
import pandas as pd

omics_data = pd.read_csv('path/to/omics_data.csv')
phenotype_data = pd.read_csv('path/to/phenotype_data.csv').squeeze()
clinical_data = pd.read_csv('path/to/clinical_data.csv')
```

Step 4: Generating Adjacency Matrix
-----------------------------------

Choose between SmCCNet or WGCNA for generating the adjacency matrix.

```python
from bioneuralnet.graph_generation import SmCCNet

smccnet = SmCCNet(config='config.yml')
adjacency_matrix = smccnet.run()
```

Step 5: Creating Embeddings
---------------------------

Generate embeddings using GNN or Node2Vec.

```python
from bioneuralnet.network_embedding import GNNEmbedding

gnn = GNNEmbedding(config='config.yml')
embeddings = gnn.run(graphs={'graph': adjacency_matrix}, node_features=omics_data)
```

Step 6: Integrating Embeddings
------------------------------

Integrate embeddings into omics data for enhanced subject representations.

```python
from bioneuralnet.subject_representation import SubjectRepresentationEmbedding

subject_rep = SubjectRepresentationEmbedding(config='config.yml')
enhanced_omics = subject_rep.run(
    adjacency_matrix=adjacency_matrix,
    omics_data=omics_data,
    phenotype_data=phenotype_data,
    clinical_data=clinical_data,
    embeddings=embeddings
)
```

Step 7: Saving Enhanced Omics Data
----------------------------------

```python
enhanced_omics.to_csv('path/to/enhanced_omics_data.csv', index=False)
```

Congratulations! You've successfully used BioNeuralNet to integrate omics data with neural network embeddings.
```

**Repeat similar detailed content for other tutorials.**

---