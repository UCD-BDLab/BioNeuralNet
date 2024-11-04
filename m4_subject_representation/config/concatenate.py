import logging
import pandas as pd
import os

def run_method(config, input_dir, output_dir):
    """
    Concatenate embeddings with omics data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running Concatenate Method")

    try:
        # Define file paths
        embeddings_file = os.path.join(input_dir, config['subject_representation']['integrate_embeddings_into_omics_data']['embeddings_file'])
        omics_data_file = os.path.join(input_dir, config['subject_representation']['integrate_embeddings_into_omics_data']['omics_data_file'])
        integrated_data_file = os.path.join(output_dir, config['subject_representation']['integrate_embeddings_into_omics_data']['output_file'])

        # Validate input files
        if not os.path.isfile(embeddings_file):
            logger.error(f"Embeddings file not found: {embeddings_file}")
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        if not os.path.isfile(omics_data_file):
            logger.error(f"Omics data file not found: {omics_data_file}")
            raise FileNotFoundError(f"Omics data file not found: {omics_data_file}")

        # Load data
        embeddings_df = pd.read_csv(embeddings_file)
        omics_df = pd.read_csv(omics_data_file)

        # Actual implemntation....

    except Exception as e:
        logger.error(f"Error in Concatenate Method: {e}")
        raise
