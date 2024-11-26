import logging
import pandas as pd
import os

def run_method(config, input_dir, output_dir):
    """
    Create scalar representations from embeddings.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running Scalar Representation Method")

    try:
        # getting the file paths
        embeddings_file = os.path.join(input_dir, config['subject_representation']['scalar-representation']['embeddings_file'])
        scalar_representation_file = os.path.join(output_dir, config['subject_representation']['scalar-representation']['output_file'])

        # making sure input file exisits
        if not os.path.isfile(embeddings_file):
            logger.error(f"Embeddings file not found: {embeddings_file}")
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

        # loading the embeddings file
        embeddings_df = pd.read_csv(embeddings_file)

        # actual implementation of the scalar representation method

    except Exception as e:
        logger.error(f"Error in Scalar Representation Method: {e}")
        raise
