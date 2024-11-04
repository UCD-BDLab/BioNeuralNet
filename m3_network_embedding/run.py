import os
import logging
import sys
import importlib.util
import shutil
from utils import setup_logging
from utils import validate_paths, find_files

def run_network_embedding(config):
    """
    Execute the network embedding component based on the provided configuration.
    """
    # Set up component-specific logging
    output_dir = os.path.abspath(config['network_embedding']['paths']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    component_log_file = os.path.join(output_dir, 'component.log')
    setup_logging(component_log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting Network Embedding Component")

    try:
        # Extract paths
        input_dir = os.path.abspath(config['network_embedding']['paths']['input_dir'])
        output_dir = os.path.abspath(config['network_embedding']['paths']['output_dir'])

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Check if input directory is empty and copy files if necessary
        source_dir = os.path.abspath(os.path.join('m2_clustering', 'output'))
        copy_files_if_input_empty(input_dir, source_dir)

        # Since 'algorithm' key is not present in the component config, set it directly
        algorithm = "node2vec"
        logger.info(f"Selected Network Embedding Algorithm: {algorithm}")

        # Dynamically import the algorithm module
        algorithm_module = load_algorithm_module(component_number=3, algorithm=algorithm)

        # Dynamic discovery of cluster files
        cluster_files = find_files(input_dir, "cluster_*.csv")
        if not cluster_files:
            logger.error(f"No cluster CSV files found in {input_dir}")
            sys.exit(1)

        # Validate cluster files
        validate_paths(*cluster_files)

        # Run embedding for each cluster
        for cluster_file in cluster_files:
            algorithm_module.run_node2vec(cluster_file, config, output_dir)

        logger.info(f"{algorithm} Embedding completed successfully.")

    except Exception as e:
        logger.error(f"Error in Network Embedding: {e}")
        sys.exit(1)

def load_algorithm_module(component_number, algorithm):
    """
    Dynamically load the algorithm implementation module based on the algorithm name.
    """
    component_name = "network_embedding"
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

def copy_files_if_input_empty(input_dir, source_dir):
    """
    Copy cluster files from source_dir to input_dir if input_dir is empty, excluding cluster_labels.csv.
    """
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.listdir(input_dir):
        # Input directory is empty
        cluster_files = [
            f for f in os.listdir(source_dir)
            if f.startswith('cluster_') and f.endswith('.csv') and f != 'cluster_labels.csv'
        ]
        if not cluster_files:
            logging.error(f"No cluster CSV files found in source directory {source_dir}")
            sys.exit(1)
        for file in cluster_files:
            src_file = os.path.join(source_dir, file)
            dst_file = os.path.join(input_dir, file)
            shutil.copy(src_file, dst_file)
            logging.info(f"Copied {file} to {input_dir}")


def find_files(directory, pattern):
    """
    Find files in a directory matching the given pattern.
    """
    import glob
    pattern = os.path.join(directory, pattern)
    return glob.glob(pattern)
