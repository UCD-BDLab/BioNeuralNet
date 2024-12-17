import os
import logging
import pkg_resources

logger = logging.getLogger(__name__)


def validate_paths(*paths: str) -> None:
    """
    Validate that all specified paths exist.

    This function checks whether each provided path exists in the filesystem. If any path does not exist,
    it logs an error message and raises a `FileNotFoundError`. If all paths are valid, it logs an informational
    message confirming successful validation.

    Args:
        *paths (str): 
            Variable length argument list of paths to validate. Each path should be a string representing
            the absolute or relative path to a file or directory.

    Raises:
        FileNotFoundError: 
            If any of the specified paths do not exist.

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

    This function uses `pkg_resources` to locate the R script file packaged within the 
    `bioneuralnet.graph_generation` module. It ensures that the script is accessible and returns
    its absolute path.

    Args:
        script_name (str): 
            The name of the R script file to retrieve (e.g., 'SmCCNet.R', 'WGCNA.R').

    Returns:
        str: 
            The absolute file path to the specified R script.

    Raises:
        FileNotFoundError: 
            If the specified R script is not found within the `graph_generation` package.
    
    """
    try:
        # Retrieve the absolute path to the R script within the 'graph_generation' package
        script_path = pkg_resources.resource_filename('bioneuralnet.graph_generation', script_name)
        if not os.path.isfile(script_path):
            logger.error(f"R script '{script_name}' not found in 'bioneuralnet.graph_generation'.")
            raise FileNotFoundError(f"R script '{script_name}' not found in 'bioneuralnet.graph_generation'.")
        logger.debug(f"Retrieved R script '{script_name}' at '{script_path}'.")
        return script_path
    except KeyError:
        logger.error(f"R script '{script_name}' does not exist in the 'graph_generation' package.")
        raise FileNotFoundError(f"R script '{script_name}' does not exist in the 'graph_generation' package.")
    except Exception as e:
        logger.error(f"An error occurred while retrieving R script '{script_name}': {e}")
        raise e


# # Module-level variables for R scripts
# try:
#     smccnet_r: str = get_r_script('SmCCNet.R')
#     logger.info(f"SmCCNet.R script located at: {smccnet_r}")
# except FileNotFoundError as fnf_error:
#     logger.error(fnf_error)
#     smccnet_r = ""

# try:
#     wgcna_r: str = get_r_script('WGCNA.R')
#     logger.info(f"WGCNA.R script located at: {wgcna_r}")
# except FileNotFoundError as fnf_error:
#     logger.error(fnf_error)
#     wgcna_r = ""
