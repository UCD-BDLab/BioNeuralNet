import os
import pandas as pd
from typing import List


def combine_omics_data(
    omics_file_paths: List[str], 
    output_path: str, 
    sample_id_col: str = 'SampleID'
) -> pd.DataFrame:
    """
    Combines multiple omics data files into a single consolidated DataFrame and saves it to a specified path.
    
    This function reads multiple omics data files (e.g., proteomics, metabolomics), ensures they share a common
    set of sample IDs, and concatenates them horizontally to create a unified dataset. The combined data is
    then saved as a CSV file to the specified output path.
    
    Args:
        omics_file_paths (List[str]): 
            A list of file paths to the individual omics data CSV files.
        output_path (str): 
            The file path where the combined omics data CSV will be saved.
        sample_id_col (str, optional): 
            The name of the column that contains the sample IDs. Defaults to 'SampleID'.
    
    Returns:
        pd.DataFrame: 
            A pandas DataFrame containing the combined omics data with samples as rows and features as columns.
    
    Raises:
        FileNotFoundError: 
            If any of the specified omics data files do not exist.
        ValueError: 
            If the specified sample ID column is not found in any of the omics data files.
        pd.errors.EmptyDataError: 
            If any of the omics data files are empty.
    
    Example:
        ```python
        from bioneuralnet.utils.data_utils import combine_omics_data

        omics_file_paths = [
            './input/proteomics_data.csv',
            './input/metabolomics_data.csv'
        ]

        combined_omics_file = './input/omics_data.csv'
        combined_omics_data = combine_omics_data(omics_file_paths, combined_omics_file)
        ```
    """
    omics_data_list: List[pd.DataFrame] = []
    
    for file_path in omics_file_paths:
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Omics file not found: {file_path}")
        
        # Read the omics data CSV file
        try:
            omics_data = pd.read_csv(file_path)
            self_print = print  # Alias to avoid issues if 'print' is overridden elsewhere
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"The omics file is empty: {file_path}")
        
        # Check if the sample ID column exists
        if sample_id_col not in omics_data.columns:
            raise ValueError(f"Sample ID column '{sample_id_col}' not found in {file_path}")
        
        # Set the sample ID column as the index for alignment
        omics_data.set_index(sample_id_col, inplace=True)
        omics_data_list.append(omics_data)
        print(f"Loaded {file_path} with shape {omics_data.shape}")
    
    # Concatenate all omics data horizontally, aligning on the sample IDs
    try:
        combined_omics_data = pd.concat(omics_data_list, axis=1, join='inner')
    except ValueError as e:
        raise ValueError(f"Error concatenating omics data: {e}")
    
    # Reset the index to include the sample ID column in the DataFrame
    combined_omics_data.reset_index(inplace=True)
    
    # Save the combined omics data to the specified output path
    try:
        combined_omics_data.to_csv(output_path, index=False)
        print(f"Combined omics data saved to {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save combined omics data to {output_path}: {e}")
    
    return combined_omics_data
