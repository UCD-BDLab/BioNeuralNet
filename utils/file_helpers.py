import os
import glob

def find_files(directory, pattern):
    """
    Find all files in a directory matching the given glob pattern.
    """
    search_pattern = os.path.join(directory, pattern)
    return glob.glob(search_pattern)

