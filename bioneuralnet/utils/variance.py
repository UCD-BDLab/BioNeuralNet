import pandas as pd
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

def variance_summary(df: pd.DataFrame, low_var_threshold: float = None) -> dict:
    """
    Compute summary statistics for column variances in the DataFrame
    """

    variances = df.var()
    summary = {
        "variance_mean": variances.mean(),
        "variance_median": variances.median(),
        "variance_min": variances.min(),
        "variance_max": variances.max(),
        "variance_std": variances.std()
    }
    if low_var_threshold is not None:
        summary["num_low_variance_features"] = (variances < low_var_threshold).sum()
    
    return summary

def zero_fraction_summary(df: pd.DataFrame, high_zero_threshold: float = None) -> dict:
    """
    Compute summary statistics for the fraction of zeros in each column
    """

    zero_fraction = (df == 0).sum(axis=0) / df.shape[0]
    summary = {
        "zero_fraction_mean": zero_fraction.mean(),
        "zero_fraction_median": zero_fraction.median(),
        "zero_fraction_min": zero_fraction.min(),
        "zero_fraction_max": zero_fraction.max(),
        "zero_fraction_std": zero_fraction.std()
    }
    if high_zero_threshold is not None:
        summary["num_high_zero_features"] = (zero_fraction > high_zero_threshold).sum()
    
    return summary

def expression_summary(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the mean expression of features
    """

    mean_expression = df.mean()

    summary = {
        "expression_mean": mean_expression.mean(),
        "expression_median": mean_expression.median(),
        "expression_min": mean_expression.min(),
        "expression_max": mean_expression.max(),
        "expression_std": mean_expression.std()
    }

    return summary

def correlation_summary(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics of the maximum pairwise correlation
    """
    corr_matrix = df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = corr_matrix.max()

    summary = {
        "max_corr_mean": max_corr.mean(),
        "max_corr_median": max_corr.median(),
        "max_corr_min": max_corr.min(),
        "max_corr_max": max_corr.max(),
        "max_corr_std": max_corr.std()
    }
    return summary

def explore_data_stats(omics_df: pd.DataFrame, name: str = "Data") -> None:
    """
    Print key statistics for an omics DataFrame including variance, zero fraction,
    """
    print(f"Statistics for {name}:")
    var_stats = variance_summary(omics_df, low_var_threshold=1e-4)
    print(f"Variance Summary: {var_stats}")
    
    zero_stats = zero_fraction_summary(omics_df, high_zero_threshold=0.50)
    print(f"Zero Fraction Summary: {zero_stats}")
    
    expr_stats = expression_summary(omics_df)
    print(f"Expression Summary: {expr_stats}")
    
    try:
        corr_stats = correlation_summary(omics_df)
        print(f"Correlation Summary: {corr_stats}")
    except Exception as e:
        print(f"Correlation Summary: Could not compute due to: {e}")
    print("\n")


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
