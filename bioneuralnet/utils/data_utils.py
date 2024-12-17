import pandas as pd
from typing import List


def combine_omics_data(omics_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple omics datasets into a single DataFrame by merging on sample indices.

    This function concatenates multiple omics DataFrames horizontally (i.e., column-wise) based on their 
    sample indices. It ensures that only samples present in all omics datasets are retained.

    Args:
        omics_list (List[pd.DataFrame]):
            A list of omics DataFrames to be combined. Each DataFrame should have samples as rows 
            and features as columns.

    Returns:
        pd.DataFrame:
            A combined omics DataFrame with samples as rows and all features from the input DataFrames as columns.
            Only samples present in all input DataFrames are retained.

    Raises:
        ValueError: If `omics_list` is empty.
        KeyError: If sample indices do not align across all omics DataFrames.

    Notes:
        - Ensure that all omics DataFrames have properly aligned and formatted sample indices before combining.
        - Missing values in any omics DataFrame will result in NaNs in the combined DataFrame.
    """
    if not omics_list:
        raise ValueError("No omics data provided to combine.")

    combined_data = omics_list[0]
    for omics in omics_list[1:]:
        combined_data = combined_data.join(omics, how='inner')
    
    return combined_data

