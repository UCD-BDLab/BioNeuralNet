import os
import logging
import sys
from utils import setup_logging
import importlib.util

# Not ready yet, will be implemented in the review...
# Current code is generic and similar to other components/modules

def run_subject_representation(config):
    """
    Execute the subject representation component based on the provided configuration.
    """
    # Set up component-specific logging
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    component_log_file = os.path.join(output_dir, 'component.log')
    setup_logging(component_log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting Subject Representation Component")

    try:
        # getting the paths from config file
        input_dir = config['paths']['input_dir']
        output_dir = config['paths']['output_dir']

        # making sure the input dir exists
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)

        # get the method to use
        method = config['subject_representation']['method']
        logger.info(f"Selected Integration Method: {method}")

        # importing the method module based on the method
        method_module = load_method_module(component_number=4, method=method)

        # execute the method for subject representation
        method_module.run_method(config, input_dir, output_dir)
        logger.info(f"{method} Method executed successfully.")

    except Exception as e:
        logger.error(f"Error in Subject Representation Component: {e}")
        sys.exit(1)

def load_method_module(component_number, algorithm):
    """
    Dynamically load the implementation module based on the algorithm name.
    """
    component_name = "subject_representation"
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
