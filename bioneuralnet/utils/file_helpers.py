import os
import glob
from typing import List


def find_files(directory: str, pattern: str) -> List[str]:
    """
    Find all files in a directory matching the given glob pattern.

    This function searches the specified directory for files that match the provided glob pattern
    and returns a list of their absolute file paths.

    Args:
        directory (str): 
            The directory to search in. Must be a valid and existing directory path.
        pattern (str): 
            The glob pattern to match files (e.g., '*.csv', 'data_*.xlsx').

    Returns:
        List[str]: 
            A list of absolute file paths that match the given pattern within the specified directory.
            Returns an empty list if no files match the pattern.

    Raises:
        NotADirectoryError: 
            If the specified `directory` does not exist or is not a directory.
        ValueError: 
            If the `pattern` is an empty string.

    Example:
        ```python
        from bioneuralnet.utils.file_helpers import find_files

        # Find all CSV files in the input directory
        input_directory = './data/input'
        csv_files = find_files(input_directory, '*.csv')
        print(csv_files)  # Outputs: ['./data/input/data1.csv', './data/input/data2.csv']
        ```
    """
    # Validate that the provided directory exists and is indeed a directory
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"The specified directory does not exist or is not a directory: {directory}")

    # Validate that the pattern is not empty
    if not pattern:
        raise ValueError("The glob pattern must be a non-empty string.")

    # Construct the full search pattern
    search_pattern = os.path.join(directory, pattern)

    # Use glob to find all files matching the pattern
    matched_files = glob.glob(search_pattern)

    # Convert matched file paths to absolute paths for consistency
    absolute_matched_files = [os.path.abspath(file_path) for file_path in matched_files]

    return absolute_matched_files
