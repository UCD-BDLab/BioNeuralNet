# BioNeuralNet

BioNeuralNet is a **modular**, **flexible**, and **extensible** bioinformatics pipeline designed to streamline the analysis of multi-omics data. It facilitates a seamless workflow encompassing graph generation, clustering, network embedding, subject representation, and task optimization. Tailored for scientists and researchers, BioNeuralNet simplifies complex computational tasks, enabling efficient data processing and insightful biological discoveries.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Pipeline Components](#pipeline-components)
  - [1. Graph Generation](#1-graph-generation)
  - [2. Clustering](#2-clustering)
  - [3. Network Embedding](#3-network-embedding)
  - [4. Subject Representation](#4-subject-representation)
  - [5. Task Optimization](#5-task-optimization)
- [Configuration](#configuration)
  - [Root Configuration](#root-configuration)
  - [Component Configuration](#component-configuration)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Entire Pipeline](#running-the-entire-pipeline)
  - [Running Individual Components](#running-individual-components)
- [Workflow Overview](#workflow-overview)
  - [Data Flow and Dependencies](#data-flow-and-dependencies)
- [Logging](#logging)
- [Extending the Pipeline](#extending-the-pipeline)
  - [Adding New Algorithms](#adding-new-algorithms)
- [Best Practices](#best-practices)
- [License](#license)
- [Contact](#contact)

## Introduction

BioNeuralNet is a **modular** and **flexible** network embedding framework tailored for bioinformatics applications. Designed as a comprehensive toolkit, BioNeuralNet streamlines the processes of network generation, embedding, clustering, and analysis, enabling scientists to extract meaningful insights from complex multi-omics data. By allowing users to run individual components or the entire pipeline sequentially, BioNeuralNet offers unparalleled adaptability to meet diverse research needs.

### Key Benefits:

- **Modularity:** Each component operates independently, facilitating easy updates and maintenance.
- **Flexibility:** Users can customize the pipeline by selecting specific components or running the full workflow.
- **Extensibility:** Seamlessly integrate new algorithms and methods without disrupting existing functionalities.
- **Usability:** User-friendly configuration through YAML files simplifies setup and execution.
- **Scalability:** Efficiently handles large datasets, making it suitable for extensive bioinformatics analyses.

## Features

- **Modular Architecture:** Each analysis step—network generation, embedding, clustering, and optimization—is encapsulated within its own module, allowing for independent development and testing.
- **Dynamic Algorithm Integration:** Effortlessly switch between different algorithms or introduce new ones by updating configuration files, without altering the core pipeline scripts.
- **Comprehensive Logging System:** Maintain detailed logs for both the overall pipeline (`pipeline.log`) and individual components (`component.log`), aiding in monitoring and debugging.
- **Reusable Helper Functions:** Streamlined file operations and data handling through centralized helper utilities, promoting code reuse and consistency.
- **Consistent Naming Conventions:** Standardized output file naming ensures clarity and prevents overwriting of results.
- **Robust Error Handling:** Implemented validation checks and exception handling mechanisms ensure the pipeline fails gracefully, providing informative error messages.
- **User-Friendly Configuration:** Centralized and component-specific YAML configuration files simplify the setup process, making the pipeline accessible to users with varying levels of technical expertise.
- **Scalability and Performance:** Optimized to handle large-scale multi-omics datasets efficiently, making it suitable for extensive bioinformatics research.

## Pipeline Architecture

![BioNeuralNet Diagram](assets/BioNeuralNet-wb.png)

BioNeuralNet's architecture is meticulously designed to promote clarity and efficiency. Each component is encapsulated within its own directory, containing all necessary scripts, configurations, and dependencies. The pipeline orchestrates these components in a sequential manner, ensuring data flows seamlessly from one analysis stage to the next.

### Architecture Overview

1. **Graph Generation:** Constructs a global network from multi-omics and phenotype data.
2. **Clustering:** Segments the global graph into meaningful clusters.
3. **Network Embedding:** Generates embeddings for each sub-network using specified embedding techniques.
4. **Subject Representation:** Integrates network embeddings with omics data to create comprehensive subject profiles.
5. **Task Optimization:** Performs predictive modeling based on subject representations to optimize downstream tasks.

## Pipeline Components

### 1. Graph Generation (`m1_graph_generation`)

**Purpose:** Constructs a global graph from multi-omics and phenotype data using specified algorithms.

- **Input:** Multi-Omics Data Files, Phenotype File.
- **Output:** `global_network.csv` saved in `m1_graph_generation/output/`.
- **Algorithms:** Implemented in `config/smccnet.py` and can be extended with additional algorithms.

### 2. Clustering (`m2_clustering`)

**Purpose:** Segments the global graph into sub-networks using clustering methods.

- **Input:** `global_network.csv` from `m1_graph_generation/output/`.
- **Output:** `cluster_1.csv`, `cluster_2.csv`, etc., saved in `m2_clustering/output/`.
- **Algorithms:** Hierarchical Clustering (`config/hierarchical.py`), PageRank Clustering (`config/pagerank.py`).

### 3. Network Embedding (`m3_network_embedding`)

**Purpose:** Generates embeddings for each sub-network using network embedding techniques.

- **Input:** `cluster_1.csv`, `cluster_2.csv`, etc., from `m2_clustering/output/`.
- **Output:** `cluster_1_embeddings.csv`, `cluster_2_embeddings.csv`, etc., saved in `m3_network_embedding/output/`.
- **Algorithms:** Node2Vec (`config/node2vec.py`).

### 4. Subject Representation (`m4_subject_representation`)

**Purpose:** Integrates network embeddings with omics data to create comprehensive subject representations.

- **Input:** Embeddings from `m3_network_embedding/output/`, Raw Multi-Omics Data.
- **Output:** `integrated_data.csv` saved in `m4_subject_representation/output/`.
- **Methods:** Concatenation (`config/concatenate.py`), Scalar Representation (`config/scalar_representation.py`).

### 5. Task Optimization (`m5_task_optimization`)

**Purpose:** Performs predictive modeling based on subject representations to optimize downstream tasks.

- **Input:** `integrated_data.csv` from `m4_subject_representation/output/`.
- **Output:** `predictions_<timestamp>.csv` saved in `m5_task_optimization/output/`.
- **Algorithms:** Random Forest, SVM, Logistic Regression (`config/prediction.py`).

## Configuration

### Root Configuration

The root `config.yml` centralizes the configuration settings for the entire pipeline. It specifies which algorithms to use for each component and general pipeline settings.

**Example `config.yml`:**

```yaml
# General pipeline settings
pipeline:
  run_all: true
  output_dir: "log_output"
  log_file: "pipeline.log"

# Graph Generation Settings
m1_graph_generation:
  algorithm: "smccnet"

# Clustering Settings
m2_clustering:
  algorithm: "Hierarchical"

# Network Embedding Settings
m3_network_embedding:
  algorithm: "node2vec"

# Subject Representation Settings
m4_subject_representation:
  integration_method: "scalar-representation"

# Task Optimization Settings
m5_task_optimization:
  task_type: "prediction"
  algorithm: "RandomForest"
```
### Component Configuration

Each component has its own `config.yml` file located in its respective `config/` directory. These files contain algorithm-specific parameters.

**Example for Clustering (`m2_clustering/config/config.yml`):**

```yaml
clustering:
  paths:
    input_dir: "../input"
    output_dir: "../output"

  Hierarchical:
    n_clusters: 5
    linkage: "ward"
    affinity: "euclidean"

  PageRank:
    damping_factor: 0.85
    max_iter: 100
    tol: 1e-6
```
**Example for Network Embedding (`m3_network_embedding/config/config.yml`):**

```yaml
network_embedding:
  paths:
    input_dir: "../input"
    output_dir: "../output"

  node2vec:
    embedding_dim: 128
    walk_length: 80
    num_walks: 10
    window_size: 10
```
## Installation

**Clone the Repository:**

```bash
git clone https://github.com/bdlab-ucd/BioNeuralNet.git
cd BioNeuralNet
```
**Set Up a Virtual Environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```
**Install Required Packages:**

```bash
pip install -r requirements.txt
```

## Usage

BioNeuralNet is designed with user-friendliness and flexibility in mind. Whether you're looking to execute the entire pipeline or focus on specific analysis steps, BioNeuralNet accommodates your research needs seamlessly.

### Running the Entire Pipeline

Execute all components sequentially to process your multi-omics data from network generation to task optimization.

```bash
python main.py --start 1 --end 5
```
**Description:**

- `--start 1`: Begins execution from Component 1 (Graph Generation).
- `--end 5`: Ends execution at Component 5 (Task Optimization).

### Running Individual Components

Focus on specific components without executing the entire pipeline. Useful for testing or when certain stages are already completed.

**Example**: Run only Component 3 (Network Embedding).

```bash
python main.py --start 3 --end 3
```

Note: Ensure that the `input_dir` of the component contains the necessary input files.

## Workflow Overview
### Data Flow and Dependencies

1. **Graph Generation**(`m1_graph_generation`):
    - **Input**: Multi-Omics Data Files, Phenotype File.
    - **Output**: `global_graph.csv` saved in `m1_graph_generation/output/`.

2. **Clustering** (`m2_clustering`):
    - **Input**: `global_graph.csv` from `m1_graph_generation/output/`.
    - **Output**: `cluster1.csv`, `cluster2.csv`, etc., saved in `m2_clustering/output/`.

3. **Network Embedding** (`m3_network_embedding`):
    - **Input**: `cluster1.csv`, `cluster2.cs`v, etc., from `m2_clustering/output/`.
    - **Output**: `node_embeddings_cluster1.csv`, `node_embeddings_cluster2.csv`, etc., saved in `m3_network_embedding/output`/.

4. **Subject Representation** (`m4_subject_representation`):
    - **Input**: `node_embeddings_cluster1.csv`, `node_embeddings_cluster2.csv`, etc., `from m3_network_embedding/output/`, Raw Multi-Omics Data.
    - **Output**: integrated_data.csv saved in `m4_subject_representation/output/`.

5. **Task Optimization** (`m5_task_optimization`):
    - **Input**: integrated_data.csv from `m4_subject_representation/output/`.
    - **Output**: predictions_<timestamp>.csv saved in `m5_task_optimization/output/`.

## Logging

BioNeuralNet implements comprehensive logging to facilitate monitoring and debugging.

- **Root Log** (`pipeline.log`): Located in the global `output_dir`, records high-level pipeline execution details.
- **Component Logs** (`component.log`): Located within each component's `output/` directory, contain detailed logs specific to that component's execution.

**Example Locations:**

- `global_output/pipeline.log`
- `m2_clustering/output/component.log`

## Extending the Pipeline

BioNeuralNet's **modular** and **extensible** design allows researchers to incorporate new algorithms and methods effortlessly. This flexibility ensures that the pipeline can evolve alongside advancements in bioinformatics and network analysis.

### Adding New Algorithms

Integrate new algorithms into BioNeuralNet by following these streamlined steps:

1. **Create the Algorithm Script:**
   - Navigate to the target component's `config/` directory.
   - Add a new Python script named after the algorithm (e.g., `new_algorithm.py`).

2. **Implement the Algorithm:**
   - Define a `run_method` function (or appropriately named) that encapsulates the algorithm's logic.
   - Ensure the function accepts necessary parameters such as `config`, `input_dir`, and `output_dir`.

**Example: `m2_clustering/config/new_algorithm.py`**

```python
import logging
import pandas as pd
from sklearn.cluster import KMeans

def run_method(network_file, config, output_dir):
    """
    Perform K-Means Clustering on the network data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running K-Means Clustering")

    try:
        # Load network data
        network_df = pd.read_csv(network_file, index_col=0)
        feature_matrix = network_df.values

        # Initialize the clustering model
        model = KMeans(
            n_clusters=config['clustering']['KMeans']['n_clusters'],
            random_state=config['clustering']['KMeans']['random_state']
        )

        # Fit the model
        labels = model.fit_predict(feature_matrix)

        # Save cluster labels
        cluster_labels_file = os.path.join(output_dir, "cluster_labels_kmeans.csv")
        cluster_labels_df = pd.DataFrame({
            'node': network_df.index,
            'cluster': labels
        })
        cluster_labels_df.to_csv(cluster_labels_file, index=False)
        logger.info(f"K-Means cluster labels saved to {cluster_labels_file}")

    except Exception as e:
        logger.error(f"Error in K-Means Clustering: {e}")
        raise
```
**3. Update Component Configuration:**

- Modify the component's `config.yml` to include the new algorithm and its specific parameters.

**Example: `m2_clustering/config/config.yml`**

```yaml
clustering:
  paths:
    input_dir: "../input"
    output_dir: "../output"

  Hierarchical:
    n_clusters: 5
    linkage: "ward"
    affinity: "euclidean"

  PageRank:
    damping_factor: 0.85
    max_iter: 100
    tol: 1e-6

  KMeans:
    n_clusters: 5
    random_state: 42
```
**4. Update Root Configuration:**

- Specify the new algorithm in the root `config.yml` for the respective component. In the example below we specify the new clustering algorithm.

**Example:**
```yaml
# Clustering Settings
m2_clustering:
  algorithm: "KMeans"
```

**5. Run the Pipeline:**

- Execute the pipeline or the specific component to utilize the new algorithm.

```bash
python main.py --start 2 --end 2
```

### Best Practices for Extending code base:

*   **Consistent Naming:** Follow standardized naming conventions for new scripts and modules to ensure seamless integration.

*   **Documentation:** Update relevant sections of the `README.md` and any in-code documentation to reflect the addition of new algorithms.

*   **Testing:** After adding a new algorithm, perform both unit tests and integration tests to verify functionality and compatibility.

*   **Version Control:** Commit changes systematically using Git, documenting the integration steps for future reference.

## License

[MIT License](/LICENSE)

## Contact

For questions or support, please contact:

- Vicente Ramos
  - Email: vicente.ramos@ucdenver.edu

- Big Data Management and Mining Laboratory
  - Email: bdlab@ucdenver.edu

