import os
import sys
import logging
import shutil
import importlib.util
from utils import setup_logging
from utils import validate_paths, find_files

def run_clustering(config):
    """
    Execute the clustering component based on the provided configuration.
    """
    # Set up component-specific logging
    output_dir = os.path.abspath(config['clustering']['paths']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    component_log_file = os.path.join(output_dir, 'component.log')
    setup_logging(component_log_file)

    logger = logging.getLogger(__name__)
    logger.info("Starting Clustering Component")

    try:
        # Extract paths
        input_dir = os.path.abspath(config['clustering']['paths']['input_dir'])
        output_dir = os.path.abspath(config['clustering']['paths']['output_dir'])

        # Ensure input directory exists
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)

        # Set the algorithm directly
        algorithm = "hierarchical"
        logger.info(f"Selected Clustering Algorithm: {algorithm}")

        # Dynamically import the algorithm module
        algorithm_module = load_algorithm_module(component_number=2, algorithm=algorithm)

        # Find the network file
        network_file = os.path.join(input_dir, 'global_network.csv')
        if not os.path.isfile(network_file):
            logger.error(f"Network file not found: {network_file}")
            sys.exit(1)

        # Validate paths
        validate_paths(input_dir, output_dir, network_file)

        # Execute the algorithm
        algorithm_module.run_hierarchical(network_file, config, output_dir)
        logger.info(f"{algorithm.capitalize()} Clustering completed successfully.")

    except Exception as e:
        logger.error(f"Error in Clustering Component: {e}")
        sys.exit(1)

def copy_files_if_input_empty(input_dir, source_dir):
    """
    Copy files from source_dir to input_dir if input_dir is empty.
    """
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.listdir(input_dir):
        # Input directory is empty
        csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        if not csv_files:
            logging.error(f"No CSV files found in source directory {source_dir}")
            sys.exit(1)
        for file in csv_files:
            src_file = os.path.join(source_dir, file)
            dst_file = os.path.join(input_dir, file)
            shutil.copy(src_file, dst_file)
            logging.info(f"Copied {file} to {input_dir}")

def load_algorithm_module(component_number, algorithm):
    """
    Dynamically load the algorithm implementation module based on the algorithm name.
    """
    component_name = "clustering"
    module_name = algorithm.lower()
    module_path = os.path.join(f"m{component_number}_{component_name}", "config", f"{module_name}.py")

    if not os.path.isfile(module_path):
        logging.error(f"Algorithm module not found: {module_path}")
        raise FileNotFoundError(f"Algorithm module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        logging.error(f"Could not load spec for module: {module_name}")
        raise ImportError(f"Could not load spec for module: {module_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module