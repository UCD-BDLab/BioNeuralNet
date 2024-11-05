import os
import logging
import sys
from utils import setup_logging
from utils import validate_paths
import importlib.util

def run_graph_generation(config):
    """
    Execute the graph generation component based on the provided configuration.
    """
    # Set up component-specific logging
    output_dir = os.path.abspath(config['graph_generation']['paths']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    component_log_file = os.path.join(output_dir, 'component.log')
    setup_logging(component_log_file)

    logger = logging.getLogger(__name__)
    logger.info("Starting Graph Generation Component")

    try:
        # Extract paths
        input_dir = os.path.abspath(config['graph_generation']['paths']['input_dir'])
        output_dir = os.path.abspath(config['graph_generation']['paths']['output_dir'])

        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)

        # Check if input directory is empty
        if not os.listdir(input_dir):
            logger.error(f"Input directory is empty: {input_dir}")
            print(f"The input directory '{input_dir}' is empty. Please add omics-data files.")
            sys.exit(1)

        # # Ensure input directory exists
        # if not os.path.isdir(input_dir):
        #     logger.error(f"Input directory does not exist: {input_dir}")
        #     sys.exit(1)

        # Since 'algorithm' key is not present, directly set to 'smccnet'
        algorithm = "smccnet"
        logger.info(f"Selected Graph Generation Algorithm: {algorithm}")

        # Dynamically import the algorithm module
        algorithm_module = load_algorithm_module(component_number=1, algorithm=algorithm)

        # Find required omics files
        required_omics = config['graph_generation']['smccnet']['omics_files']
        omics_file_paths = [os.path.join(input_dir, f) for f in required_omics]

        # Check for missing omics files
        missing_omics = [f for f in required_omics if not os.path.isfile(os.path.join(input_dir, f))]
        if missing_omics:
            logger.error(f"Missing required omics files: {missing_omics}")
            sys.exit(1)

        # Find phenotype file
        phenotype_file = config['graph_generation']['smccnet']['phenotype_file']
        phenotype_file_path = os.path.join(input_dir, phenotype_file)

        if not os.path.isfile(phenotype_file_path):
            logger.error(f"Phenotype file not found: {phenotype_file_path}")
            sys.exit(1)

        # Validate all required files and directories
        validate_paths(
            os.path.abspath(config['graph_generation']['paths']['input_dir']),
            os.path.abspath(config['graph_generation']['paths']['output_dir']),
            *omics_file_paths,
            phenotype_file_path,
            os.path.abspath(config['graph_generation']['smccnet']['saving_dir']) if 'saving_dir' in config['graph_generation']['paths'] else output_dir
        )

        # Execute the algorithm
        algorithm_module.run_smccnet(omics_file_paths, phenotype_file_path, config, output_dir)
        logger.info(f"{algorithm} Algorithm executed successfully.")

    except Exception as e:
        logger.error(f"Error in Graph Generation Component: {e}")
        sys.exit(1)

def load_algorithm_module(component_number, algorithm):
    """
    Dynamically load the algorithm implementation module based on the algorithm name.
    """
    component_name = "graph_generation"
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
