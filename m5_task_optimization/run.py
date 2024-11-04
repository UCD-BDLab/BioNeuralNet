import os
import logging
import sys
from utils import setup_logging
import importlib.util

# Not ready yet, will be implemented in the review...
# Current code is generic and similar to other components/modules

def run_task_optimization(config):
    """
    Execute the task optimization component based on the provided configuration.
    """
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    component_log_file = os.path.join(output_dir, 'component.log')
    setup_logging(component_log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting Task Optimization Component")

    try:
        input_dir = config['paths']['input_dir']
        output_dir = config['paths']['output_dir']

        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)

        task_type = config['task_optimization']['task_type']
        algorithm = config['task_optimization']['algorithm']
        logger.info(f"Selected Task Type: {task_type}")
        logger.info(f"Selected Prediction Algorithm: {algorithm}")

        if task_type != "prediction":
            logger.error(f"Unsupported Task Type: {task_type}")
            sys.exit(1)

        # importing the prediction module based on the algorithm
        prediction_module = load_prediction_module(component_number=5, algorithm=algorithm)

        # executing the algorithm
        prediction_module.run_method(config, input_dir, output_dir)
        logger.info(f"{algorithm} Prediction Task executed successfully.")

    except Exception as e:
        logger.error(f"Error in Task Optimization Component: {e}")
        sys.exit(1)

def load_prediction_module(component_number, algorithm):
    """
    Dynamically load the prediction implementation module based on the algorithm name.
    """
    component_name = "task_optimization"
    module_name = algorithm.lower()
    module_path = os.path.join(f"{component_number}_{component_name}", "config", f"{module_name}.py")

    if not os.path.isfile(module_path):
        logging.error(f"Prediction module not found: {module_path}")
        raise FileNotFoundError(f"Prediction module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        logging.error(f"Could not load spec for module: {module_name}")
        raise ImportError(f"Could not load spec for module: {module_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

