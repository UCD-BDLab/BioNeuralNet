import os
import logging
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_paths(*paths: str) -> None:
    """
    Validate that all specified paths exist.

    Args:
        *paths (str): Variable length argument list of paths to validate.

    Raises:
        FileNotFoundError: If any of the specified paths do not exist.
    """
    for path in paths:
        if not os.path.exists(path):
            logger.error(f"Path '{path}' does not exist.")
            raise FileNotFoundError(f"Path '{path}' does not exist.")
        else:
            logger.debug(f"Path '{path}' exists.")
    logger.info("All paths validated successfully.")


def get_r_script(script_name: str) -> str:
    """
    Retrieve the absolute path to an R script within the `graph_generation` package.

    Args:
        script_name (str): The name of the R script file to retrieve (e.g., 'SmCCNet.R', 'WGCNA.R').

    Returns:
        str: The absolute file path to the specified R script.

    Raises:
        FileNotFoundError: If the specified R script is not found within the `graph_generation` package.
    """
    try:
        with resources.path(
            "bioneuralnet.graph_generation", script_name
        ) as script_path:
            script_path = script_path.resolve()
            if not script_path.is_file():
                logger.error(
                    f"R script '{script_name}' not found in 'bioneuralnet.graph_generation'."
                )
                raise FileNotFoundError(
                    f"R script '{script_name}' not found in 'bioneuralnet.graph_generation'."
                )
            logger.debug(f"Retrieved R script '{script_name}' at '{script_path}'.")
            return str(script_path)
    except FileNotFoundError:
        logger.error(
            f"R script '{script_name}' does not exist in the 'graph_generation' package."
        )
        raise
    except Exception as e:
        logger.error(
            f"An error occurred while retrieving R script '{script_name}': {e}"
        )
        raise
