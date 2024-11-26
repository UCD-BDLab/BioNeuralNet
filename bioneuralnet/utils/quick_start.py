# import os
# import yaml
# from typing import Dict, Any


# def quick_start() -> None:
#     """
#     Initialize the BioNeuralNet project by generating a default configuration file,
#     creating necessary directories, and setting up an interface guide.
    
#     This function performs the following actions:
#         1. Generates a default `config.yml` file at the root of the project with predefined settings.
#         2. Creates the main directories: `./logs`, `./input`, and `./output`.
#         3. Creates component-specific subdirectories within the `./output` directory.
#         4. Generates an `interface.txt` file in the `./input` directory to guide users on necessary input files.
    
#     Raises:
#         Exception: If there is an error writing the configuration file or creating directories.
    
#     Example:
#         ```python
#         from bioneuralnet.utils.quick_start import quick_start

#         quick_start()
#         ```
#     """
#     try:
#         # Step 1: Create default config.yml
#         default_config: Dict[str, Any] = {
#             'paths': {
#                 'input_dir': './input',
#                 'output_dir': './output',
#                 'log_dir': './logs'
#             },
#             'smccnet': {
#                 # Default parameters for SmCCNet
#                 'omics_file': 'omics_data.csv',
#                 'phenotype_file': 'phenotype_data.csv',
#                 'data_types': 'protein, metabolite',  # Comma-separated string of data types
#                 'kfold': 5,
#                 'summarization': 'PCA',
#                 'seed': 732
#             },
#             'wgcna': {
#                 # Default parameters for WGCNA
#                 'omics_file': 'omics_data.csv',
#                 'phenotype_file': 'phenotype_data.csv',
#                 'soft_power': 6,
#                 'min_module_size': 30,
#                 'merge_cut_height': 0.25
#             },
#             'gnn_embedding': {
#                 # Default parameters for GNN Embedding
#                 'model_type': 'GCN',
#                 'gnn_hidden_dim': 64,
#                 'gnn_layer_num': 2,
#                 'dropout': True
#             },
#             'node2vec_embedding': {
#                 # Default parameters for Node2Vec Embedding
#                 'embedding_dim': 128,
#                 'walk_length': 80,
#                 'num_walks': 10,
#                 'window_size': 10
#             },
#             'subject_representation': {
#                 # Default parameters for Subject Representation
#                 # Add relevant parameters as needed
#             }
#         }

#         config_path: str = 'config.yml'
#         with open(config_path, 'w') as file:
#             yaml.dump(default_config, file)
#         print(f"Generated default {config_path} at the root of your project.")

#         # Step 2: Create main directories
#         main_directories: list = ['./logs', './input', './output']
#         for dir_path in main_directories:
#             os.makedirs(dir_path, exist_ok=True)
#             print(f"Ensured directory exists: {dir_path}")

#         # Step 3: Create component-specific output subdirectories
#         output_subdirs: list = [
#             './output/smccnet_output',
#             './output/gnn_output',
#             './output/node2vec_output',
#             './output/subject_representation_output'
#         ]
#         for dir_path in output_subdirs:
#             os.makedirs(dir_path, exist_ok=True)
#             print(f"Created component-specific output directory: {dir_path}")

#         # Step 4: Create interface.txt in input directory
#         interface_text: str = (
#             "Welcome to BioNeuralNet!\n\n"
#             "Please ensure the following input files are placed in the 'input' directory:\n"
#             "- Omics data files (e.g., omics_data.csv)\n"
#             "- Phenotype data files (e.g., phenotype_data.csv)\n"
#             "- Clinical data files (e.g., clinical_data.csv)\n\n"
#             "Refer to the documentation for detailed instructions on preparing these files."
#         )
#         interface_file_path: str = './input/interface.txt'
#         with open(interface_file_path, 'w') as f:
#             f.write(interface_text)
#         print(f"Generated interface guide at {interface_file_path}")

#         print("\nQuick start setup is complete. Please move the necessary input files to the 'input' directory before running components.")

#     except Exception as e:
#         print(f"An error occurred during the quick start setup: {e}")
#         raise e
