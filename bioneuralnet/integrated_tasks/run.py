import os
import sys
import logging
import yaml
import glob
import shutil
# from ..utils.logger import get_logger
# from m5_integrated_tasks.DPMON import run_dpmon

# def run_prediction_dpmon(config):
#     """
#     Execute prediction with network information using DPMON.
#     """
#     # Set up component-specific logging
#     component_output_dir = os.path.abspath(config['integrated_tasks']['paths']['output_dir'])
#     os.makedirs(component_output_dir, exist_ok=True)
#     component_log_file = os.path.join(component_output_dir, 'component.log')
#     setup_logging(component_log_file)
#     logger = logging.getLogger(__name__)

#     logger.info("Starting Task Optimization Component using DPMON")

#     try:
#         # Extract paths from component config
#         input_dir = os.path.abspath(config['integrated_tasks']['paths']['input_dir'])
#         output_dir = os.path.abspath(config['integrated_tasks']['paths']['output_dir'])

#         # Ensure input directory exists
#         if not os.path.isdir(input_dir):
#             logger.error(f"Input directory does not exist: {input_dir}")
#             sys.exit(1)

#         # Check if input directory is empty and prompt user if necessary
#         copy_files_if_input_empty(input_dir)

#         # Extract DPMON parameters from component config
#         prediction_config = config['integrated_tasks']['prediction']
#         model = prediction_config.get('model', 'GCN')
#         gpu = prediction_config.get('gpu', False)
#         cuda = prediction_config.get('cuda', 0)
#         tune = prediction_config.get('tune', False)
#         ## we pribably will want some default here from public availble data.
#         network_file = prediction_config.get('network_file', None)
#         omics_files = prediction_config.get('omics_files', None)
#         phenotype_file = prediction_config.get('phenotype_file', None)
#         features_file = prediction_config.get('features_file', None)

#         # Default values for DPMON parameters
#         layer_num = prediction_config.get('gnn_layer_num', 3)
#         gnn_hidden_dim = prediction_config.get('gnn_hidden_dim', 128)
#         lr = prediction_config.get('lr', 0.01)
#         weight_decay = prediction_config.get('weight_decay', 1e-4)
#         nn_hidden_dim1 = prediction_config.get('nn_hidden_dim1', 128)
#         nn_hidden_dim2 = prediction_config.get('nn_hidden_dim2', 16)
#         epoch_num = prediction_config.get('num_epochs', 50)
#         repeat_num = prediction_config.get('repeat_num', 10)

#         # Ensure input directory exists
#         if not os.path.isdir(input_dir):
#             logger.error(f"Input directory does not exist: {input_dir}")
#             sys.exit(1)

#         # Check if input directory is empty and prompt user if necessary
#         copy_files_if_input_empty(input_dir)

#         # Verify that all required files exist in the input directory
#         missing_files = []
#         # Combine required files: omics_files + phenotype_file + network_file
#         required_files = [os.path.join(input_dir, file) for file in omics_files + [phenotype_file, network_file]]
#         if features_file:
#             required_files.append(os.path.join(input_dir, features_file))

#         # Check existence of each required file
#         for file_path in required_files:
#             if not os.path.isfile(file_path):
#                 missing_files.append(os.path.basename(file_path))

#         if missing_files:
#             logger.error(f"Missing input files: {', '.join(missing_files)}")
#             sys.exit(1)

#         # Prepare parameters for DPMON
#         dpmon_params = {
#             'model': model,
#             'dataset_dir': input_dir,
#             'lr': lr,
#             'weight_decay': weight_decay,
#             'layer_num': layer_num,
#             'gnn_hidden_dim': gnn_hidden_dim,
#             'nn_hidden_dim1': nn_hidden_dim1,
#             'nn_hidden_dim2': nn_hidden_dim2,
#             'epoch_num': epoch_num,
#             'repeat_num': repeat_num,
#             'network_file': os.path.join(input_dir, network_file),
#             'features_file': features_file,
#             'phenotype_file': os.path.join(input_dir, phenotype_file),
#             'omics_files': [os.path.join(input_dir, omics_file) for omics_file in omics_files],
#             'gpu': gpu,
#             'cuda': cuda,
#             'tune': tune
#         }

#         # Log the parameters being passed to DPMON
#         logger.info(f"Running DPMON with parameters: {dpmon_params}")

#         # Execute DPMON's main function directly
#         run_dpmon(dpmon_params, output_dir)

#         logger.info("Task Optimization completed successfully.")

#     except Exception as e:
#         logger.error(f"Error in Task Optimization Component: {e}")
#         sys.exit(1)

# def copy_files_if_input_empty(input_dir):
#     """
#     Copy necessary files from previous components if input_dir is empty.
#     """
#     if not os.listdir(input_dir):
#         print(f"The input directory '{input_dir}' is empty.")
#         print("Select the source of input files:")
#         print("1. Use raw input files from Component 1 (Graph Generation)")
#         print("2. Use cluster outputs from Component 2 (Clustering)")
#         print("3. Provide input files manually")
#         choice = input("Enter your choice (1/2/3): ").strip()
#         if choice == '1':
#             source_dir = os.path.abspath(os.path.join('m1_graph_generation', 'input'))
#             files_copied = copy_files(source_dir, input_dir, pattern='*.*')  # Copy all raw input files
#             if not files_copied:
#                 print(f"No files found in {source_dir}.")
#                 sys.exit(1)
#         elif choice == '2':
#             source_dir = os.path.abspath(os.path.join('m2_clustering', 'output'))
#             files_copied = copy_files(source_dir, input_dir, pattern='cluster_*.csv')  # Copy cluster files
#             if not files_copied:
#                 print(f"No cluster CSV files found in {source_dir}.")
#                 sys.exit(1)
#         elif choice == '3':
#             print("Please provide the required input files in the input directory.")
#             sys.exit(1)
#         else:
#             print("Invalid choice. Exiting.")
#             sys.exit(1)

# def copy_files(source_dir, dest_dir, pattern):
#     """
#     Copy files matching the pattern from source_dir to dest_dir.
#     """
#     files = glob.glob(os.path.join(source_dir, pattern))
#     for file in files:
#         shutil.copy(file, dest_dir)
#         print(f"Copied {os.path.basename(file)} to {dest_dir}")
#     return len(files) > 0




# if __name__ == "__main__":
#     # Load the root configuration file
#     root_config_path = os.path.join(os.getcwd(), 'config.yml')
#     if not os.path.isfile(root_config_path):
#         print(f"Root configuration file not found: {root_config_path}")
#         sys.exit(1)

#     with open(root_config_path, 'r') as file:
#         root_config = yaml.safe_load(file)

#     # Run the prediction
#     run_prediction_dpmon(root_config)
