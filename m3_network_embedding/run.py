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
        #source_dir = os.path.abspath(os.path.join('m2_clustering', 'output'))
        copy_files_if_input_empty(input_dir)

        # Since 'algorithm' key is not present in the component config, set it directly
        algorithm = "node2vec"
        logger.info(f"Selected Network Embedding Algorithm: {algorithm}")

        # Dynamically import the algorithm module
        algorithm_module = load_algorithm_module(component_number=3, algorithm=algorithm)

        # Dynamic discovery of cluster files
        input_files = find_files(input_dir, "*.csv")
        if not input_files:
            logger.error(f"No CSV files found in {input_dir}")
            sys.exit(1)

        # Exclude 'cluster_labels.csv' if present
        input_files = [f for f in input_files if not f.endswith('cluster_labels.csv')]

        # Run embedding for each input file
        for input_file in input_files:
            algorithm_module.run_node2vec(input_file, config, output_dir)

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

def copy_files_if_input_empty(input_dir):
    """
    Copy files from the appropriate source directory to input_dir if input_dir is empty,
    with user prompts.
    """
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.listdir(input_dir):
        # Input directory is empty
        print(f"The input directory '{input_dir}' is empty.")
        print("Select the source of input files:")
        print("1. Use output from Component 2 (Clustering)")
        print("2. Use output from Component 1 (Graph Generation)")
        print("3. Provide input files manually")
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == '1':
            source_dir = os.path.abspath(os.path.join('m2_clustering', 'output'))
            cluster_files = [
                f for f in os.listdir(source_dir)
                if f.startswith('cluster_') and f.endswith('.csv') and f != 'cluster_labels.csv'
            ]
            if cluster_files:
                for file in cluster_files:
                    src_file = os.path.join(source_dir, file)
                    dst_file = os.path.join(input_dir, file)
                    shutil.copy(src_file, dst_file)
                    logging.info(f"Copied {file} to {input_dir}")
                    print(f"Copied {file} from Component 2 to '{input_dir}'.")
            else:
                print(f"No cluster files found in {source_dir}. Please run Component 2 first or provide the necessary input files.")
                sys.exit(1)
        elif choice == '2':
            source_dir = os.path.abspath(os.path.join('m1_graph_generation', 'output'))
            network_file = 'global_network.csv'
            src_file = os.path.join(source_dir, network_file)
            dst_file = os.path.join(input_dir, network_file)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
                logging.info(f"Copied {network_file} to {input_dir}")
                print(f"Copied {network_file} from Component 1 to '{input_dir}'.")
            else:
                print(f"Error: {network_file} not found in {source_dir}. Please run Component 1 first or provide the necessary input files.")
                sys.exit(1)
        elif choice == '3':
            print("Please provide the required input files in the input directory.")
            sys.exit(1)
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)



def find_files(directory, pattern):
    """
    Find files in a directory matching the given pattern.
    """
    import glob
    pattern = os.path.join(directory, pattern)
    return glob.glob(pattern)
