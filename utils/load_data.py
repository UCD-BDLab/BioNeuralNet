import os
import logging

logger = logging.getLogger(__name__)

def validate_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            logger.error(f"Path '{path}' does not exist.")
            raise FileNotFoundError(f"Path '{path}' does not exist.")
    logger.info("All paths validated successfully.")
