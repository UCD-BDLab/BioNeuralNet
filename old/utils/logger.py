import logging
import sys

def setup_logging(log_file="pipeline.log"):
    """
    Set up logging for the entire pipeline.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # console handler: allows logging to stdout or git bash console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # handles the actual logging file
    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler: {e}")
