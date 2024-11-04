import logging
import pandas as pd
import os

def run_method(config, input_dir, output_dir):
    """
    Perform some downstream task using the specified algorithm.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running downstream task")

    try:
        # Defining file paths
        integrated_data_file = os.path.join(input_dir, "integrated_data.csv")
        if not os.path.isfile(integrated_data_file):
            logger.error(f"Integrated data file not found: {integrated_data_file}")
            raise FileNotFoundError(f"Integrated data file not found: {integrated_data_file}")

        # Loading the data
        data = pd.read_csv(integrated_data_file)
        logger.info(f"Loaded integrated data from {integrated_data_file}")

        # actual implementation of the some downstream application ....

    except Exception as e:
        logger.error(f"Error in downstream Task: {e}")
        raise
