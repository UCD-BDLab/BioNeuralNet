# main.py

import os
import sys
import yaml
import logging
from utils import setup_logging
from utils import get_parser
from utils import adjust_component_config, run_component, component_run_functions, component_mapping

# Import component modules with numerical prefixes
import m1_graph_generation.run as graph_generation
import m2_clustering.run as clustering
import m3_network_embedding.run as network_embedding
import m4_subject_representation.run as subject_representation
import m5_integrated_tasks.run as integrated_tasks


component_run_functions[1] = graph_generation.run_graph_generation
component_run_functions[2] = clustering.run_clustering
component_run_functions[3] = network_embedding.run_network_embedding
component_run_functions[4] = subject_representation.run_subject_representation
component_run_functions[5] = integrated_tasks.run_integrated_tasks

def load_root_config(config_path='config.yml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def display_interface():
    """
    Displays the BioNeuralNet Interface with configuration notices and available options.
    """
    interface_text = """
============================================
                BIONEURALNET INTERFACE
============================================

**Configuration Notice:**
- **Global Configuration:** Please modify the `config.yml` at the root of the pipeline to adjust general settings.
- **Component-Specific Configuration:** Each component has its own `config.yml` located at `componentname/config/config.yml`. Customize algorithm-specific parameters within these files as needed.

## Data Flow and Dependencies

1. **Graph Generation** (`m1_graph_generation`):
    - **Input**: Multi-Omics Data Files, Phenotype File.
    - **Output**: `global_graph.csv` saved in `m1_graph_generation/output/`.

2. **Clustering** (`m2_clustering`):
    - **Input**: `global_graph.csv` from `m1_graph_generation/output/`.
    - **Output**: `cluster1.csv`, `cluster2.csv`, etc., saved in `m2_clustering/output/`.

3. **Network Embedding** (`m3_network_embedding`):
    - **Input**: `cluster1.csv`, `cluster2.csv`, etc., from `m2_clustering/output/`.
    - **Output**: `node_embeddings_cluster1.csv`, `node_embeddings_cluster2.csv`, etc., saved in `m3_network_embedding/output/`.

4. **Subject Representation** (`m4_subject_representation`):
    - **Input**: `node_embeddings_cluster1.csv`, `node_embeddings_cluster2.csv`, etc., from `m3_network_embedding/output/`, Raw Multi-Omics Data.
    - **Output**: `integrated_data.csv` saved in `m4_subject_representation/output/`.

5. **Integrated Task Processing** (`m5_integrated_tasks`):
    - **Input**:
        - **Mandatory:**
            - `network_file`: An omics network file (`network.csv`). Can be from `m1_graph_generation/output/`, `m2_clustering/output/`, or a user-provided `network.csv` as an adjacency matrix.
            - `omics_files`: List of omics data files (e.g., `metabolites.csv`, `proteins.csv`).
            - `phenotype_file`: `phenotype.csv`.
        - **Optional but Recommended:**
            - `features_file`: Clinical data for nodes (e.g., `features.csv`).
    - **Output**: `predictions.csv` saved in `m5_integrated_tasks/output/`.
    - **Description**: Utilizes DPMON for disease prediction based on integrated multi-omics data and network information. Can incorporate additional downstream tasks for comprehensive analysis.

============================================
## Pipeline Components

- ### 1. Graph Generation (`m1_graph_generation`)

  **Purpose:** Constructs a global graph from multi-omics and phenotype data using specified algorithms.

  - **Input:** Multi-Omics Data Files, Phenotype File.
  - **Output:** `global_network.csv` saved in `m1_graph_generation/output/`.
  - **Algorithms:** Implemented in `config/smccnet.py` and can be extended with additional algorithms.

- ### 2. Clustering (`m2_clustering`)

  **Purpose:** Segments the global graph into sub-networks using clustering methods.

  - **Input:** `global_network.csv` from `m1_graph_generation/output/`.
  - **Output:** `cluster_1.csv`, `cluster_2.csv`, etc., saved in `m2_clustering/output/`.
  - **Algorithms:** Hierarchical Clustering (`config/hierarchical.py`), PageRank Clustering (`config/pagerank.py`).

- ### 3. Network Embedding (`m3_network_embedding`)

  **Purpose:** Generates embeddings for each sub-network using network embedding techniques.

  - **Input:** `cluster_1.csv`, `cluster_2.csv`, etc., from `m2_clustering/output/`.
  - **Output:** `node_embeddings_cluster1.csv`, `node_embeddings_cluster2.csv`, etc., saved in `m3_network_embedding/output/`.
  - **Algorithms:** Node2Vec (`config/node2vec.py`).

- ### 4. Subject Representation (`m4_subject_representation`)

  **Purpose:** Integrates network embeddings with omics data to create comprehensive subject representations.

  - **Input:** Embeddings from `m3_network_embedding/output/`, Raw Multi-Omics Data.
  - **Output:** `integrated_data.csv` saved in `m4_subject_representation/output/`.
  - **Methods:** Concatenation (`config/concatenate.py`), Scalar Representation (`config/scalar_representation.py`).

- ### 5. Integrated Task Processing (`m5_integrated_tasks`)

  **Purpose:** Performs disease prediction and additional downstream analyses using integrated multi-omics data and network information.

  - **Input:**
      - **Mandatory:**
          - `network_file`: An omics network file (`network.csv`) from `m1_graph_generation/output/`, `m2_clustering/output/`, or a user-provided adjacency matrix.
          - `omics_files`: List of omics data files (e.g., `metabolites.csv`, `proteins.csv`).
          - `phenotype_file`: `phenotype.csv`.
      - **Optional but Recommended:**
          - `features_file`: Clinical data for nodes (e.g., `features.csv`).
  - **Output:** `predictions.csv` saved in `m5_integrated_tasks/output/`.
  - **Description:** Utilizes DPMON for disease prediction based on integrated multi-omics data and network information. Can incorporate additional downstream tasks for comprehensive analysis.

============================================
            BioNeuralNet Main Menu
============================================
Select the components you want to run:
1. Graph Generation
2. Clustering
3. Network Embedding
4. Subject Representation
5. Integrated Task Processing

Enter the numbers of the components separated by commas (e.g., 1,3):


============================================
    """
    print(interface_text)

def main():
    #if there is no comamnd line argument provided by the user then display the menu
    if len(sys.argv) == 1:
        display_interface()
        user_input = input().strip()
        selected_components = [int(x) for x in user_input.split(',') if x.strip().isdigit()]

        # Load root config
        root_config = load_root_config()


        # Set up root logging
        output_dir = root_config['pipeline'].get('output_dir', 'global_output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pipeline_log = os.path.join(output_dir, root_config['pipeline'].get('log_file', 'pipeline.log'))
        setup_logging(pipeline_log)
        logger = logging.getLogger(__name__)

        logger.info("Pipeline configuration loaded from config.yml")

        # Validate selected components
        valid_components = [1, 2, 3,4, 5]
        for comp in selected_components:
            if comp not in valid_components:
                print(f"Component {comp} is either invalid or under construction.")
                return

        # Run the selected components
        run_selected_components(selected_components, root_config, logger)

    else:
        # Existing functionality using --start and --end
        parser = get_parser()
        args = parser.parse_args()

        # Load root config
        root_config = load_root_config()

        # Set up root logging
        output_dir = root_config['pipeline'].get('output_dir', 'global_output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pipeline_log = os.path.join(output_dir, root_config['pipeline'].get('log_file', 'pipeline.log'))
        setup_logging(pipeline_log)
        logger = logging.getLogger(__name__)

        logger.info("Pipeline configuration loaded from config.yml")

        run_all = root_config['pipeline'].get('run_all', False)

        start = args.start
        end = args.end

        logger.info(f"Pipeline run_all: {run_all}")
        logger.info(f"Pipeline execution range: {start} to {end}")

        if start < 1 or start > 5 or end < 1 or end > 5:
            logger.error("Component indices must be between 1 and 5.")
            sys.exit(1)

        if start > end:
            logger.error("--start cannot be greater than --end.")
            sys.exit(1)

        selected_components = list(range(start, end + 1))
        run_selected_components(selected_components, root_config, logger)

def run_selected_components(selected_components, root_config, logger):
    previous_output_dir = None

    for component_number in selected_components:
        component_name = component_mapping[component_number]
        logger.info(f"Running component {component_number}: {component_name}")

        if component_number > 3:
            print(f"Component {component_number} ({component_name}) is under construction.")
            continue

        # Adjust config based on previous component's output if necessary
        if previous_output_dir:
            component_config = adjust_component_config(component_number, root_config, previous_output_dir)
        else:
            component_config = adjust_component_config(component_number, root_config)

        # Run the component
        run_component(component_number, component_config)

        # Update previous_output_dir for the next component
        output_dir = component_config[component_name]['paths']['output_dir']
        if output_dir:
            previous_output_dir = os.path.abspath(output_dir)
            logger.info(f"Component {component_number} completed. Outputs saved to {previous_output_dir}")
        else:
            logger.warning(f"No output_dir found for component {component_number}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error(f"An error occurred: {e}")
        sys.exit(1)