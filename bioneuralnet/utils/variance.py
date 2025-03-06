import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)

def remove_variance(df: pd.DataFrame, variance_threshold: float = 1e-6) -> pd.DataFrame:
    """
    Remove columns from the DataFrame with variance below the specified threshold.
    
    Parameters:

        df (pd.DataFrame): Input data.
        variance_threshold (float): Variance threshold.
        
    Returns:

        pd.DataFrame: DataFrame with only columns having variance above the threshold.
    """
    logger.info(f"Removing columns with variance below {variance_threshold}.")
    variances = df.var()
    filtered_df = df.loc[:, variances > variance_threshold]
    logger.info(f"Original shape: {df.shape}, Filtered shape: {filtered_df.shape}")
    return filtered_df

def remove_fraction(df: pd.DataFrame, zero_frac_threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove columns from the DataFrame where the fraction of zeros is higher than the threshold.
    
    Parameters:

        df (pd.DataFrame): Input data.
        zero_frac_threshold (float): Maximum allowed fraction of zero entries.
        
    Returns:

        pd.DataFrame: DataFrame with only columns having a lower zero fraction.
    """
    logger.info(f"Removing columns with zero fraction >= {zero_frac_threshold}.")
    zero_fraction = (df == 0).sum(axis=0) / df.shape[0]
    filtered_df = df.loc[:, zero_fraction < zero_frac_threshold]
    logger.info(f"Original shape: {df.shape}, Filtered shape: {filtered_df.shape}")
    return filtered_df

def network_remove_low_variance(network: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    """
    Remove rows and columns from adjacency matrix where the variance is below a threshold.
    
    Parameters:

        network (pd.DataFrame): Adjacency matrix.
        threshold (float): Variance threshold.
        
    Returns:

        pd.DataFrame: Filtered adjacency matrix.
    """
    logger.info(f"Removing low-variance rows/columns with threshold {threshold}.")
    variances = network.var(axis=0)
    valid_indices = variances[variances > threshold].index
    filtered_network = network.loc[valid_indices, valid_indices]
    logger.info(f"Original network shape: {network.shape}, Filtered shape: {filtered_network.shape}")
    return filtered_network

def network_remove_high_zero_fraction(network: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove rows and columns from adjacency matrix where the fraction of zero entries is higher than the threshold.
    
    Parameters:

        network (pd.DataFrame): Adjacency matrix.
        threshold (float): Zero-fraction threshold.
        
    Returns:

        pd.DataFrame: Filtered adjacency matrix.
    """
    logger.info(f"Removing high zero fraction features with threshold: {threshold}.")
    zero_fraction = (network == 0).sum(axis=0) / network.shape[0]
    valid_indices = zero_fraction[zero_fraction < threshold].index
    filtered_network = network.loc[valid_indices, valid_indices]
    logger.info(f"Original network shape: {network.shape}, Filtered shape: {filtered_network.shape}")
    return filtered_network

def network_filter(network: pd.DataFrame, threshold: float, filter_type: str = 'variance') -> pd.DataFrame:
    """
    Filter an adjacency matrix using either variance or zero fraction criteria.
    
    Parameters:

        network (pd.DataFrame): Adjacency matrix.
        threshold (float): Threshold for filtering.
        filter_type (str): Type of filter to apply; either 'variance' or 'zero_fraction'.
        
    Returns:

        pd.DataFrame: Filtered adjacency matrix.
        
    Raises:

        ValueError: If an invalid filter_type is provided.
    """
    logger.info(f"Filtering network with {filter_type} threshold of {threshold}.")
    logger.info(f"Original network shape: {network.shape}")

    if filter_type == 'variance':
        return network_remove_low_variance(network, threshold)
    elif filter_type == 'zero_fraction':
        return network_remove_high_zero_fraction(network, threshold)
    else:
        raise ValueError(f"Invalid filter type: {filter_type}. Must be 'variance' or 'zero_fraction'.")

def omics_data_filter(omics: pd.DataFrame, variance_threshold: float = 1e-6, zero_frac_threshold: float = 0.95) -> pd.DataFrame:
    """
    Filter omics data by removing columns with low variance and high zero fraction.
    
    Parameters:

        omics (pd.DataFrame): Omics data.
        variance_threshold (float): Variance threshold.
        zero_frac_threshold (float): Zero fraction threshold.
        
    Returns:
    
        pd.DataFrame: Filtered omics data.
    """
    logger.info("Filtering omics data.")
    logger.info(f"Original omics shape: {omics.shape}")
    
    filtered_omics = remove_variance(omics, variance_threshold)
    filtered_omics = remove_fraction(filtered_omics, zero_frac_threshold)
    
    logger.info(f"Final omics shape: {filtered_omics.shape}")
    return filtered_omics