"""
External Tools Module

This module provides utility functions for interoperability between Python and R.
It handles the execution of external R scripts to extract, convert, and load
RData structures (such as cross-validation folds and network matrices) into
standardized Python data structures like pandas DataFrames and NumPy arrays.

Available Functions:

* `extract_and_load_folds`: Triggers Rscript extraction and loads the folds.
* `load_r_export_folds`: Directly loads a previously extracted R directory structure.
* `rdata_to_df`: Converts an arbitrary .RData file object to a pandas DataFrame.
"""

from .extract_CVfold import extract_and_load_folds, load_r_export_folds
from .rdata_to_df import rdata_to_df

__all__ = [
    "extract_and_load_folds",
    "load_r_export_folds",
    "rdata_to_df"
]
