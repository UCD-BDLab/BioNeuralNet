# BioNeuralNet

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyPI](https://img.shields.io/pypi/v/bioneuralnet)
![Python Versions](https://img.shields.io/pypi/pyversions/bioneuralnet)

**BioNeuralNet** is designed to integrate omics data with neural network embeddings, facilitating advanced data analysis in bioinformatics. Leveraging Python, BioNeuralNet constructs and analyzes complex biological networks, ensuring comprehensive and scalable data processing.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
    - [For Users: Python Package Installation](#for-users-python-package-installation)
    - [For Users: R Dependencies](#for-users-r-dependencies)
    - [For Developers: Development Environment Setup](#for-developers-development-environment-setup)
4. [Quick Start Guide](#quick-start-guide)
5. [Pipeline Components](#pipeline-components)
6. [Acknowledgements](#acknowledgements)
7. [Documentation](#documentation)
8. [Testing](#testing)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)


## Overview

BioNeuralNet empowers researchers to seamlessly integrate various omics datasets using advanced neural network embeddings. By combining the strengths of Python for data manipulation and analysis, BioNeuralNet provides a comprehensive toolkit for bioinformatics applications.


![BioNeuralNet Workflow](/assets/BioNeuralNet.png)


## Features

- **Graph Construction**: Utilize SmCCNet and WGCNA algorithms to build complex biological networks.
- **Clustering**: Perform clustering using methods like PageRank and Hierarchical Clustering.
- **Network Embedding**: Generate high-dimensional embeddings using Graph Neural Networks (GNNs) and Node2Vec.
- **Subject Representation**: Enhance omics data representations for downstream analyses.
- **Comprehensive Testing**: Ensure reliability with a robust testing suite and continuous integration.
- **Developer-Friendly**: Streamlined setup for contributors with pre-commit hooks and development tools.


## Installation

### Python Package Installation

Most users can install **BioNeuralNet** directly via `pip`, which includes all necessary Python dependencies.

1. **Ensure Python 3.7 or Higher is Installed**

   Verify your Python version:

   ```bash
   python3 --version
   ```

2. **Install BioNeuralNet via Pip**

   ```bash
   pip install bioneuralnet
   ```

   *This command installs the latest stable release from PyPI, including all base dependencies.*

   **Note:** If you require CUDA-enabled functionalities (for GPU acceleration), ensure that you have the appropriate CUDA version installed on your system.

### R Dependencies

**BioNeuralNet** integrates R scripts for graph construction using SmCCNet and WGCNA. While Python users can install the package via `pip`, R dependencies need separate installation.

#### 1. Manual R Installation

If you prefer manual installation or are on an unsupported operating system, follow these steps:

**a. Install R**

- **Download R:**

   Visit the [CRAN R Project](https://cran.r-project.org/) and download the appropriate installer for your operating system.

- **Install R:**

  Follow the installation instructions provided on the CRAN website for your specific OS.

**b. Install Required R Packages**

Open R or RStudio and execute the following commands to install necessary packages:

```R
install.packages(c("dplyr", "SmCCNet", "WGCNA"))
```

**Notes:**

- **System Dependencies:** Some R packages might require additional system dependencies. Ensure you have the necessary build tools installed (e.g., `gcc`, `make`).

- **Permissions:** You may need administrative privileges to install certain packages or dependencies.

### Fast Install: Auto install R and development dependencies

Fastes way to get everything up an running is using the script `fast-install.py`. This involves creating a virtual environment, installing dependencies (including R), and setting up pre-commit hooks to maintain code quality.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/UCD-BDLab/BioNeuralNet.git
   cd BioNeuralNet
   ```

2. **Run the Setup Script**

   The `fast-install.py` script automates the setup process.

   ```bash
   cd scripts
   python3 fast-install.py
   ```

   **What `fast-install.py` Does:**

   - **Creates and Activates a Virtual Environment:** Ensures that dependencies are isolated.
   - **Installs Base and Development Dependencies:** Sets up the environment with necessary packages.
   - **Installs Pre-Commit Hooks:** Automates code quality checks before commits.

   **Note:** Ensure you have execution permissions for `setup-env.sh` and `setup-R.sh`. If not, make it executable:

   ```bash
   chmod +x setup-env.sh
   chmod +x setup-R.sh
   ```
**Note**: Steps 3 and 4 are only necessary if you plan to contribute to the codebase.

3. **Verify Pre-Commit Hooks**

   After running `fast-install.py`, pre-commit hooks should be installed automatically. To confirm, you can run:

   ```bash
   pre-commit run --all-files
   ```

   *This will execute all configured pre-commit hooks on the entire codebase.*

4. **Install R Dependencies (If Needed)**

   If your development work involves R scripts, ensure that R and its required packages are installed as per the [R Dependencies](#for-users-r-dependencies) section.



## Quick Start Guide

Begin using **BioNeuralNet** by following these streamlined steps:

1. **Prepare Input Data**

   - **Input Directory:** Create an `input/` directory at the root of your project.
   - **Omics Data:** Place your omics CSV files (e.g., `proteomics_data.csv`, `metabolomics_data.csv`) in the `input/` directory.
   - **Phenotype Data:** Place `phenotype_data.csv` in the `input/` directory.
   - **Clinical Data:** Place `clinical_data.csv` in the `input/` directory.

2. **Combine Omics Data**, if needed.

   If you have multiple omics datasets, combine them using the `combine_omics_data` utility.

   ```python
   from bioneuralnet.utils.data_utils import combine_omics_data

   omics_file_paths = [
       './input/proteomics_data.csv',
       './input/metabolomics_data.csv'
   ]

   combined_omics_file = './input/omics_data.csv'

   combine_omics_data(omics_file_paths, combined_omics_file)
   ```

3. **Examples**

   Execute of BioNeuralNet components

   3.1. **SmCCnet**
   ```Python
   from bioneuralnet.graph_generation.smccnet import SmCCNet

   def main():
      smccnet = SmCCNet(
         phenotype_file='input/phenotype_data.csv',
         omics_list=[
               'input/proteins.csv',
               'input/metabolites.csv'
         ],
         data_types=['protein', 'metabolite'],  
         kfold=5,                              
         summarization='PCA',                  
         seed=732,                              
      )

      adjacency_matrix = smccnet.run()
      adjacency_matrix.to_csv('smccnet_output_1/global_network.csv')

      print("Adjacency Matrix saved to 'smccnet_output_1/global_network.csv'.")
    ```
   3.2. **WGCNA**
   ```Python
   from bioneuralnet.graph_generation.wgcna import WGCNA

   def main():
      wgcna = WGCNA(
         phenotype_file='input/phenotype_data.csv',
         omics_list=[
               'input/genes.csv',
               'input/miRNA.csv'
         ],
         # Default values for WGCNA parameters
         data_types=['gene', 'miRNA'],        
         soft_power=6,                        
         min_module_size=30,                  
         merge_cut_height=0.25,                
         output_dir='wgcna_output_1'           
      )

      # Run WGCNA to generate adjacency matrix
      adjacency_matrix = wgcna.run()

      # Save the adjacency matrix to a CSV file
      adjacency_matrix.to_csv('wgcna_output_1/global_network.csv')

      print("Adjacency Matrix saved to 'wgcna_output_1/global_network.csv'.")

   if __name__ == "__main__":
      main()
    ```

   3.3. **Generate Embeddings using GNNs**
   ```Python
   import pandas as pd
   from bioneuralnet.network_embedding.gnns import GNNEmbedding

   def main():
      # Paths to input files
      omics_files = ['input/proteins.csv', 'input/metabolites.csv']
      phenotype_file = 'input/phenotype_data.csv'
      clinical_data_file = 'input/clinical_data.csv' 
      adjacency_matrix_file = 'input/adjacency_matrix.csv'

      # Load adjacency matrix
      adjacency_matrix = pd.read_csv(adjacency_matrix_file, index_col=0)

      # Initialize GNNEmbedding
      gnn_embedding = GNNEmbedding(
         omics_list=omics_files,
         phenotype_file=phenotype_file,
         clinical_data_file=clinical_data_file,
         adjacency_matrix=adjacency_matrix,
         model_type='GCN', 
         gnn_hidden_dim=64,
         gnn_layer_num=2,
         dropout=True
      )

      # Run GNN embedding to generate embeddings
      embeddings_dict = gnn_embedding.run()

      # Access embeddings
      embeddings_tensor = embeddings_dict['graph']
      embeddings_df = pd.DataFrame(
         embeddings_tensor.numpy(),
         index=adjacency_matrix.index,
         columns=[f"dim_{i}" for i in range(embeddings_tensor.shape[1])]
      )

      print("Embeddings generated successfully.")
      print("Sample embeddings:")
      print(embeddings_df.head())

   if __name__ == "__main__":
      main()
    ```

   3.4. **Generate Embeddings using Node2Vec**
   ```Python
   from bioneuralnet.network_embedding.node2vec import Node2VecEmbedding

   def main():
      # Initialize Node2VecEmbedding parameters
      node2vec_embedding = Node2VecEmbedding(
         input_dir='input/graphs/',    
         embedding_dim=128,            
         walk_length=80,              
         num_walks=10,                 
         window_size=10,              
         workers=4,                   
         seed=42,                      
         output_dir=None               
      )

      # Run Node2Vec to generate embeddings
      embeddings = node2vec_embedding.run()

      # embeddings for a specific graph
      graph_name = 'global_network' 
      if graph_name in embeddings:
         embeddings_df = embeddings[graph_name]
         print(f"Embeddings for '{graph_name}' generated successfully.")
      else:
         print(f"No embeddings found for '{graph_name}'.")

   if __name__ == "__main__":
      main()
    ```
   3.5. **Subject Representation**
   ```Python
   import pandas as pd
   from bioneuralnet.subject_representation.subject_representation import SubjectRepresentationEmbedding

   def main():
      # Paths to input files
      omics_files = ['input/proteins.csv', 'input/metabolites.csv']
      phenotype_file = 'input/phenotype_data.csv'
      clinical_data_file = 'input/clinical_data.csv'
      adjacency_matrix_file = 'input/adjacency_matrix.csv'

      # Load adjacency matrix
      adjacency_matrix = pd.read_csv(adjacency_matrix_file, index_col=0)

      # Initialize SubjectRepresentationEmbedding
      subject_rep_embedding = SubjectRepresentationEmbedding(
         adjacency_matrix=adjacency_matrix,
         omics_list=omics_files,
         phenotype_file=phenotype_file,
         clinical_data_file=clinical_data_file,
         embedding_method='GNNs'
      )

      # Run the subject representation process
      enhanced_omics_data = subject_rep_embedding.run()

      # The enhanced omics data is saved to the output directory specified in the class
      print("Subject representation workflow completed successfully.")


   if __name__ == "__main__":
      main()

    ```
   3.6. **Hierarchical Clustering**
   ```Python
   from bioneuralnet.clustering.hierarchical import HierarchicalClustering

   def main():
      # Initialize HierarchicalClustering parameters
      hierarchical_cluster = HierarchicalClustering(
         adjacency_matrix_file='input/global_network.csv',
         n_clusters=3,            
         linkage='ward',         
         affinity='euclidean',   
      )

      # Run the hierarchical clustering
      results = hierarchical_cluster.run()

      # Access results
      cluster_labels_df = results['cluster_labels']
      print("Cluster labels:")
      print(cluster_labels_df.head())

      silhouette_score = results['silhouette_score']
      print(f"Silhouette Score: {silhouette_score}")

   if __name__ == "__main__":
      main()

   ```
   3.7. **PageRank Clustering**
   ```Python
   from bioneuralnet.clustering.pagerank import PageRankClustering

   def main():
      # Initialize PageRankClustering parameters
      pagerank_cluster = PageRankClustering(
         graph_file='input/GFEV1ac110.edgelist',
         omics_data_file='input/X.xlsx',
         phenotype_data_file='input/Y.xlsx',
         alpha=0.9,
         max_iter=100,
         tol=1e-6,
         k=0.9,
      )

      # Define seed nodes 
      seed_nodes = [94] 

      # Run PageRank clustering
      results = pagerank_cluster.run(seed_nodes=seed_nodes)

      # Access results
      cluster_nodes = results['cluster_nodes']
      print(f"Identified cluster with {len(cluster_nodes)} nodes.")

   if __name__ == "__main__":
      main()
   ```

4. **Hybrid Examples**

   The example aboves are uses of individual BioNeuralNet components. Here are 2 additional examples using multiple components.

   4.1. **Enhanced Subject Representation using SmCCNet and GNN embeddings.**
   ```python
   import os
   import pandas as pd

   from bioneuralnet.graph_generation.smccnet import SmCCNet
   from bioneuralnet.network_embedding.gnns import GNNEmbedding
   from bioneuralnet.subject_representation.subject_representation import SubjectRepresentationEmbedding


   def run_smccnet_workflow():
      """
      Executes the SmCCNet-based workflow for generating enhanced omics data.

      This function performs the following steps:
         1. Instantiates the SmCCNet, GNNEmbedding, and SubjectRepresentationEmbedding components.
         2. Loads omics, phenotype, and clinical data.
         3. Generates an adjacency matrix using SmCCNet.
         4. Computes node features based on correlations.
         5. Generates embeddings using GNNEmbedding.
         6. Reduces embeddings using PCA.
         7. Integrates embeddings into omics data to produce enhanced omics data.
         8. Saves the enhanced omics data to the output directory.
      """
      try:
         # Step 1: Instantiate SmCCNet parameters
         smccnet_instance = SmCCNet(
               phenotype_file='input/phenotype_data.csv',
               omics_list=[
                  'input/proteins.csv',
                  'input/metabolites.csv'
               ],
               data_types=['protein', 'metabolite'],
               kfold=5,
               summarization='PCA',
               seed=732,
         )

         # Step 2: Load omics, phenotype, and clinical data
         omics_data = pd.read_csv('input/omics_data.csv', index_col=0)
         phenotype_data = pd.read_csv('input/phenotype_data.csv', index_col=0).squeeze()
         clinical_data = pd.read_csv('input/clinical_data.csv', index_col=0)

         # Step 3: Generate adjacency matrix using SmCCNet
         adjacency_matrix = smccnet_instance.run()

         # Save adjacency matrix
         # Output dir is automatically generated by SmCCNet
         adjacency_output_path = os.path.join(smccnet_instance.output_dir, 'adjacency_matrix.csv')
         adjacency_matrix.to_csv(adjacency_output_path)
         print(f"Adjacency matrix saved to {adjacency_output_path}")

         # Step 4: Compute node features based on correlations
         subject_rep = SubjectRepresentationEmbedding()
         node_phenotype_corr = subject_rep.compute_node_phenotype_correlation(omics_data, phenotype_data)
         node_clinical_corr = subject_rep.compute_node_clinical_correlation(omics_data, clinical_data)
         node_features = pd.concat([node_clinical_corr, node_phenotype_corr.rename('phenotype_corr')], axis=1)

         # Step 5: Generate embeddings using GNNEmbedding
         gnn_embedding = GNNEmbedding(
               input_dir='', 
               model_type='GCN',
               gnn_input_dim=node_features.shape[1],
               gnn_hidden_dim=64,
               gnn_layer_num=2,
               dropout=True,
         )
         embeddings_dict = gnn_embedding.run(graphs={'graph': adjacency_matrix}, node_features=node_features)
         embeddings_tensor = embeddings_dict['graph']
         embeddings_df = pd.DataFrame(embeddings_tensor.numpy(), index=node_features.index)

         # Step 6: Reduce embeddings using PCA
         node_embedding_values = subject_rep.reduce_embeddings(embeddings_df)

         # Step 7: Integrate embeddings into omics data
         enhanced_omics_data = subject_rep.run(
               adjacency_matrix=adjacency_matrix,
               omics_data=omics_data,
               phenotype_data=phenotype_data,
               clinical_data=clinical_data,
               embeddings=node_embedding_values
         )

         # Step 8: Save the enhanced omics data
         enhanced_omics_output_path = os.path.join(subject_rep.output_dir, 'enhanced_omics_data.csv')
         enhanced_omics_data.to_csv(enhanced_omics_output_path)
         print(f"Enhanced omics data saved to {enhanced_omics_output_path}")

      except FileNotFoundError as fnf_error:
         print(f"File not found error: {fnf_error}")
         raise fnf_error
      except Exception as e:
         print(f"An error occurred during the SmCCNet workflow: {e}")
         raise e

   if __name__ == "__main__":
    try:
        print("Starting SmCCNet and GNNs Workflow...")
        run_smccnet_workflow()
        print("SmCCNet Workflow completed successfully.\n")

    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e

   ```

   4.2. **Enhanced Subject Representation using WGCNA and GNN embeddings**
   ```python
   import os
   import pandas as pd

   from bioneuralnet.graph_generation.wgcna import WGCNA
   from bioneuralnet.network_embedding.gnns import GNNEmbedding
   from bioneuralnet.network_embedding.node2vec import Node2VecEmbedding
   from bioneuralnet.subject_representation.subject_representation import SubjectRepresentationEmbedding


   def run_wgcna_workflow():
      """
      Executes the WGCNA-based workflow for generating enhanced omics data.

      This function performs the following steps:
         1. Instantiates the WGCNA, GNNEmbedding, and SubjectRepresentationEmbedding components.
         2. Loads omics, phenotype, and clinical data.
         3. Generates an adjacency matrix using WGCNA.
         4. Computes node features based on correlations.
         5. Generates embeddings using GNNEmbedding.
         6. Reduces embeddings using PCA.
         7. Integrates embeddings into omics data to produce enhanced omics data.
         8. Saves the enhanced omics data to the output directory.
      """
      try:
         # Step 1: Instantiate WGCNA with direct parameters
         wgcna_instance = WGCNA(
               phenotype_file='input/phenotype_data.csv',
               omics_list=['input/omics_data.csv'],
               data_types=['gene'],  # Adjust based on your data
               soft_power=6,
               min_module_size=30,
               merge_cut_height=0.25,
         )

         # Step 2: Load omics, phenotype, and clinical data
         omics_data = pd.read_csv('input/omics_data.csv', index_col=0)
         phenotype_data = pd.read_csv('input/phenotype_data.csv', index_col=0).squeeze()
         clinical_data = pd.read_csv('input/clinical_data.csv', index_col=0)

         # Step 3: Generate adjacency matrix using WGCNA
         adjacency_matrix = wgcna_instance.run()

         # Save adjacency matrix
         adjacency_output_path = os.path.join(wgcna_instance.output_dir, 'adjacency_matrix.csv')
         adjacency_matrix.to_csv(adjacency_output_path)
         print(f"Adjacency matrix saved to {adjacency_output_path}")

         # Step 4: Compute node features based on correlations
         subject_rep = SubjectRepresentationEmbedding()
         node_phenotype_corr = subject_rep.compute_node_phenotype_correlation(omics_data, phenotype_data)
         node_clinical_corr = subject_rep.compute_node_clinical_correlation(omics_data, clinical_data)
         node_features = pd.concat([node_clinical_corr, node_phenotype_corr.rename('phenotype_corr')], axis=1)

         # Step 5: Generate embeddings using GNNEmbedding
         gnn_embedding = GNNEmbedding(
               input_dir='',  # Not used because we pass graphs directly
               model_type='GCN',
               gnn_input_dim=node_features.shape[1],
               gnn_hidden_dim=64,
               gnn_layer_num=2,
               dropout=True,
         )
         embeddings_dict = gnn_embedding.run(graphs={'graph': adjacency_matrix}, node_features=node_features)
         embeddings_tensor = embeddings_dict['graph']
         embeddings_df = pd.DataFrame(embeddings_tensor.numpy(), index=node_features.index)

         # Step 6: Reduce embeddings using PCA
         node_embedding_values = subject_rep.reduce_embeddings(embeddings_df)

         # Step 7: Integrate embeddings into omics data
         enhanced_omics_data = subject_rep.run(
               adjacency_matrix=adjacency_matrix,
               omics_data=omics_data,
               phenotype_data=phenotype_data,
               clinical_data=clinical_data,
               embeddings=node_embedding_values
         )

         # Step 8: Save the enhanced omics data
         enhanced_omics_output_path = os.path.join(subject_rep.output_dir, 'enhanced_omics_data.csv')
         enhanced_omics_data.to_csv(enhanced_omics_output_path)
         print(f"Enhanced omics data saved to {enhanced_omics_output_path}")

      except FileNotFoundError as fnf_error:
         print(f"File not found error: {fnf_error}")
         raise fnf_error
      except Exception as e:
         print(f"An error occurred during the WGCNA workflow: {e}")
         raise e

   if __name__ == "__main__":
      try:
         print("Starting WGCNA and GNNs Workflow...")
         run_wgcna_workflow()
         print("WGCNA Workflow completed successfully.\n")
         
      except Exception as e:
         print(f"An error occurred during the execution: {e}")
         raise e
   ```

5. **Feature Selector**

   Identifies and prioritizes the most relevant multi-omics features associated with a specific phenotype. Leveraging embeddings generated by Graph Neural Networks (GNNs), this component ensures that selected genetic features capture complex interactions and are highly predictive of the outcomes of interest.

   #### **Purpose**

   Selecting key genetic features is crucial for enhancing model performance, reducing computational complexity, and improving interpretability. By utilizing network embeddings, Feature Selector captures intricate relationships within the data, leading to more meaningful feature selection.

   #### **Supported Feature Selection Methods**

   - **Correlation-Based (`'correlation'`):** Utilizes ANOVA (`f_classif`) to select genetic features that show significant association with the phenotype.
   - **LASSO-Based (`'lasso'`):** Employs LASSO regression (`LassoCV`) to identify genetic features with non-zero coefficients, indicating their importance.
   - **Random Forest-Based (`'random_forest'`):** Uses feature importances derived from a Random Forest classifier to select the top contributing genetic features.

   #### **Usage Example**

   ```python
   import pandas as pd
   from analysis.feature_selector import FeatureSelector
   from analysis.subject_representation_embedding import SubjectRepresentationEmbedding

   def main():
      # Paths to input files
      omics_files = ['input/genetic_data.csv', 'input/protein_data.csv', 'input/metabolite_data.csv']  # Replace with your omics data files
      phenotype_file = 'input/phenotype_data.csv'  # Columns: 'Asthma'
      clinical_data_file = 'input/clinical_data.csv'  # Additional clinical information
      adjacency_matrix_file = 'input/adjacency_matrix.csv'  # Feature interaction network

      # Load adjacency matrix
      adjacency_matrix = pd.read_csv(adjacency_matrix_file, index_col=0)

      # Initialize and run SubjectRepresentationEmbedding
      subject_rep = SubjectRepresentationEmbedding(
         adjacency_matrix=adjacency_matrix,
         omics_list=omics_files,
         phenotype_file=phenotype_file,
         clinical_data_file=clinical_data_file,
         embedding_method='GNNs',  # Options: 'GNNs', 'Node2Vec'
      )
      enhanced_omics_data = subject_rep.run()

      # Load phenotype data
      phenotype_data = pd.read_csv(phenotype_file, index_col=0).iloc[:, 0]

      # Initialize and run FeatureSelector
      feature_selector = FeatureSelector(
         enhanced_omics_data=enhanced_omics_data,
         phenotype_data=phenotype_data,
         num_features=20,  # Number of top features to select
         selection_method='lasso',  # Options: 'correlation', 'lasso', 'random_forest'
      )
      selected_genetic_features = feature_selector.run_feature_selection()

      # Display selected features
      print("Selected Multi-Omics Features:")
      print(selected_genetic_features.head())

   if __name__ == "__main__":
      main()

   ```

## Pipeline Components

BioNeuralNet's pipeline consists of several interconnected components:

1. **Graph Construction**

   - **SmCCNet**: Builds graphs based on higher-order correlations.
   - **WGCNA**: Constructs weighted correlation networks and detects modules.

2. **Clustering**

   - **PageRankClustering**: Clusters nodes based on personalized PageRank.
   - **HierarchicalClustering**: Performs agglomerative hierarchical clustering.

3. **Network Embedding Generation**

   - **GNNEmbedding**: Generates embeddings using Graph Neural Networks.
   - **Node2VecEmbedding**: Generates embeddings using the Node2Vec algorithm.

4. **Subject Representation**

   - **SubjectRepresentationEmbedding**: Integrates node embeddings into omics data to enhance subject representations.

   ![Subject Representation Workflow](/assets/Subject.png)

5. **Integrated Tasks**

   - **Disease Prediction using Multi-Omics Networks**: Leverages the power of Graph Neural Networks (GNNs) to capture intricate relationships between biological entities and extract valuable knowledge from this network structure.

   ![DPMON](/assets/DPMON.png)

6. **Utility Functions**

   - Includes tools for data manipulation, path validation, and more.


## Acknowledgements

BioNeuralNet utilizes several external packages and libraries that are integral to its functionality. We extend our gratitude to the developers and contributors of these projects:

- **SmCCNet**
  - *Description*: An R package for Sparse Multiple Canonical Correlation Network.
  - *Repository*: [SmCCNet on CRAN](https://cran.r-project.org/package=SmCCNet)

- **WGCNA**
  - *Description*: Weighted Correlation Network Analysis for R.
  - *Repository*: [WGCNA on CRAN](https://cran.r-project.org/package=WGCNA)

- **Node2Vec**
  - *Description*: Scalable Feature Learning for Networks.
  - *Repository*: [Node2Vec GitHub](https://github.com/aditya-grover/node2vec)

- **Other Libraries**
  - **dplyr**: A grammar of data manipulation for R. [dplyr on CRAN](https://cran.r-project.org/package=dplyr)
  - **PyTorch**: An open source machine learning library based on the Torch library. [PyTorch Official Site](https://pytorch.org/)
  - **PyTorch Geometric**: Extension library for PyTorch to handle geometric data. [PyTorch Geometric GitHub](https://github.com/pyg-team/pytorch_geometric)
  - **Pytest**: A framework that makes building simple and scalable tests easy. [Pytest Official Site](https://pytest.org/)
  - **Pre-commit**: A framework for managing and maintaining multi-language pre-commit hooks. [Pre-commit GitHub](https://github.com/pre-commit/pre-commit)
  - **Other Dependencies**: Refer to `requirements.txt` for a complete list of Python dependencies.

*Thank you to all the open-source communities that make projects like BioNeuralNet possible.*


## Documentation

Comprehensive documentation is available to help you navigate and utilize all features of **BioNeuralNet**.

- **Main Documentation:** Located in the [`docs/`](docs/) directory, providing detailed guides and usage instructions.
- **API Reference:** Detailed API documentation is available [here](https://yourdocumentationurl.com/api). *(Replace with your actual documentation URL)*
- **Testing Documentation:** Refer to [`tests/README.md`](tests/README.md) for information on running and writing tests.
- **Additional Guides:** Explore other `README.md` files within subdirectories for specific component details.

*Note:* The **README.md** provides an overview and essential instructions, while the **API Reference** offers in-depth technical details about the package's classes, functions, and methods.


## Testing

Ensuring the reliability of **BioNeuralNet** is paramount. Automated tests run on every commit and pull request via GitHub Actions, and pre-commit hooks enforce local testing before code is committed.

### Overview

- **Testing Framework:** Utilizes `pytest` for writing and running tests.
- **Continuous Integration:** GitHub Actions runs tests on multiple Python versions, checks code quality, and reports coverage.
- **Local Enforcement:** Pre-commit hooks automate tests and code quality checks before commits.

### Running Tests Locally

Detailed instructions are available in [`tests/README.md`](tests/README.md), but here's a quick overview:

1. **Ensure Development Dependencies are Installed**

   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run All Tests**

   ```bash
   pytest
   ```

3. **View Coverage Reports**

   ```bash
   pytest --cov=bioneuralnet --cov-report=html tests/
   ```

   *Open `htmlcov/index.html` in your browser to view the coverage details.*

### Continuous Integration

Every commit and pull request triggers GitHub Actions workflows that:

- Install dependencies.
- Lint and format code.
- Run the test suite.
- Upload coverage reports to Codecov.

Ensure that all tests pass in the CI pipeline before merging changes.

---

## Contributing

Contributions are welcome! To ensure a smooth collaboration process, please adhere to the following guidelines:

1. **Fork the Repository**

   ```bash
   git clone https://github.com/UCD-BDLab/BioNeuralNet.git
   cd BioNeuralNet
   ```

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install Development Dependencies**

   ```bash
   ./setup.sh
   ```

4. **Make Your Changes**

   - Write clean, readable code following PEP 8 standards.
   - Add or update tests as necessary.
   - Update documentation if your changes affect usage or functionality.

5. **Run Tests Locally**

   ```bash
   pytest
   ```

6. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Add feature XYZ"
   ```

   *Pre-commit hooks will run automatically, ensuring code quality and passing tests before the commit is finalized.*

7. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request**

   Navigate to the original repository and open a pull request detailing your changes.



## Pre-Commit Hooks

To maintain code quality, pre-commit hooks are enforced. After setting up your development environment, ensure that pre-commit hooks are installed:

```bash
pre-commit install
```

*These hooks will automatically run tests, format code, and perform linting before each commit.*

## Guidelines for Writing Tests

Refer to [`tests/README.md`](tests/README.md) for comprehensive guidelines on writing effective and consistent tests.

## License

This project is licensed under the [MIT License](LICENSE).


## Contact

For questions, support, or contributions, please open an issue on [GitHub](https://github.com/UCD-BDLab/BioNeuralNet/issues) or contact the maintainers directly.

---