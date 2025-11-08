import os
import random
import torch
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from .logger import get_logger

logger = get_logger(__name__)

def variance_summary(df: pd.DataFrame, low_var_threshold: Optional[float] = None) -> dict:
    """
    Computes key summary statistics for the feature (column) variances within an omics DataFrame.

    This is useful for assessing feature distribution and identifying low-variance features prior to modeling.

    Args:

        df (pd.DataFrame): The input omics DataFrame (samples as rows, features as columns).
        low_var_threshold (Optional[float]): A threshold used to count features falling below this variance level.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and standard deviation of the column variances. If a threshold is provided, it also includes 'num_low_variance_features'.
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

def zero_fraction_summary(df: pd.DataFrame, high_zero_threshold: Optional[float] = None) -> dict:
    """
    Computes statistics on the fraction of zero values present in each feature (column).

    This helps identify feature sparsity, which is common in omics data (e.g., RNA-seq FPKM).

    Args:

        df (pd.DataFrame): The input omics DataFrame.
        high_zero_threshold (Optional[float]): A threshold used to count features whose zero-fraction exceeds this value.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and standard deviation of the zero fractions across all columns. If a threshold is provided, it includes 'num_high_zero_features'.
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
    Computes summary statistics for the mean expression (average value) of all features.

    Provides insight into the overall magnitude and central tendency of the data values.

    Args:

        df (pd.DataFrame): The input omics DataFrame.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and standard deviation of the feature means.
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
    Computes summary statistics on the maximum pairwise (absolute) correlation observed for each feature in the DataFrame.

    This helps identify features that are highly redundant or collinear.

    Args:

        df (pd.DataFrame): The input omics DataFrame.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and standard deviation of the maximum absolute correlation values.
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
    Prints a comprehensive set of key statistics for an omics DataFrame.

    Combines variance, zero fraction, expression, and correlation summaries for rapid data quality assessment.

    Args:

        omics_df (pd.DataFrame): The input omics DataFrame.
        name (str): A descriptive name for the dataset (e.g., 'X_rna_final') for clear output labeling.

    Returns:

        None: Prints the statistics directly to the console.
    """

    logger.info(f"Statistics for {name}:")
    var_stats = variance_summary(omics_df, low_var_threshold=1e-4)
    logger.info(f"Variance Summary: {var_stats}")

    zero_stats = zero_fraction_summary(omics_df, high_zero_threshold=0.50)
    logger.info(f"Zero Fraction Summary: {zero_stats}")

    expr_stats = expression_summary(omics_df)
    logger.info(f"Expression Summary: {expr_stats}")

    try:
        corr_stats = correlation_summary(omics_df)
        logger.info(f"Correlation Summary: {corr_stats}")

    except Exception as e:
        logger.info(f"Correlation Summary: Could not compute due to: {e}")

    logger.info("\n")

def impute_omics(omics_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Imputes missing values (NaNs) in the omics DataFrame using a specified strategy.

    Args:

        omics_df (pd.DataFrame): The input DataFrame containing missing values.
        method (str): The imputation strategy to use. Must be 'mean', 'median', or 'zero'.

    Returns:

        pd.DataFrame: The DataFrame with missing values filled.
        
    Raises:

        ValueError: If the specified imputation method is not recognized.
    """
    if method == "mean":
        return omics_df.fillna(omics_df.mean())
    elif method == "median":
        return omics_df.fillna(omics_df.median())
    elif method == "zero":
        return omics_df.fillna(0)
    else:
        raise ValueError(f"Imputation method '{method}' not recognized.")

def impute_omics_knn(omics_df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputes missing values (NaNs) using the K-Nearest Neighbors (KNN) approach.

    KNN imputation replaces missing values with the average value from the 'n_neighbors' most similar samples/features. This is often more accurate than simple mean imputation.

    Args:

        omics_df (pd.DataFrame): The input DataFrame containing missing values (NaNs).
        n_neighbors (int): The number of nearest neighbors to consider for imputation.

    Returns:

        pd.DataFrame: The DataFrame with missing values filled using KNN.
    """
    
    has_non_numeric = False
    for col in omics_df.columns:
        if not pd.api.types.is_numeric_dtype(omics_df[col]):
            has_non_numeric = True
            break
            
    if has_non_numeric:
        logger.error("KNNImputer requires numeric data. Non-numeric columns found.")
    
    logger.info(f"Starting KNN imputation (k={n_neighbors}) on DataFrame (shape: {omics_df.shape}).")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(omics_df.values)
    imputed_df = pd.DataFrame(imputed_data, index=omics_df.index, columns=omics_df.columns)
    logger.info("KNN imputation complete")

    return imputed_df
    
def normalize_omics(omics_df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Scales or transforms feature data using common normalization techniques.

    Args:

        omics_df (pd.DataFrame): The input omics DataFrame.
        method (str): The scaling strategy. Must be 'standard' (Z-score), 'minmax', or 'log2'.

        
    Returns:

        pd.DataFrame: The DataFrame with features normalized according to the specified method.
        
    Raises:

        ValueError: If the specified normalization method is not recognized.
    """
    logger.info(f"Starting normalization on DataFrame (shape: {omics_df.shape}) using method: '{method}'.")
    data = omics_df.values
    
    if method == "standard":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == "minmax":
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == "log2":
        if np.any(data < 0):
            logger.warning("Log2 transformation applied to data containing negative values. This can lead to unpredictable results")
        scaled_data = np.log2(data + 1)
    else:
        logger.error(f"Normalization method '{method}' not recognized.")
        raise ValueError(f"Normalization method '{method}' not recognized.")

    final_df = pd.DataFrame(scaled_data, index=omics_df.index, columns=omics_df.columns)
    logger.info("Normalization complete.")
    return final_df
    
def set_seed(seed_value: int) -> None:
    """
    Sets seeds for maximum reproducibility across Python, NumPy, and PyTorch.

    This function sets global random seeds and configures PyTorch/CUDNN to use deterministic algorithms, ensuring that the experiment produces the exact same numerical result across different runs.

    Args:

        seed_value (int): The integer value to use as the random seed.

    Returns:

        None

    """
    logger.info(f"Setting global seed for reproducibility to: {seed_value}")
    
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        logger.info("CUDA available. Applying seed to all GPU operations")
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.info("CUDA not available. Seeding only CPU operations")
        
    logger.info("Seed setting complete")

def beta_to_m(df, eps=1e-6):
    """
    Converts methylation Beta-values (ratio of methylated intensity to total intensity) to M-values using log2 transformation.

    M-values follow a normal distribution, improving statistical analysis, especially for differential methylation studies, by transforming the constrained [0, 1] Beta scale to an unbounded log-transformed scale (-inf to +inf).

    Args:

        df (pd.DataFrame): The input DataFrame containing Beta-values (0 to 1).
        eps (float): A small epsilon value used to clip Beta-values (B) away from 0 and 1, preventing logarithm errors (log(0) or division by zero).

    Returns:

        pd.DataFrame: A new DataFrame containing the log2-transformed M-values, calculated as log2(B / (1 - B)).
    """
    logger.info(f"Starting Beta-to-M value conversion (shape: {df.shape}). Epsilon: {eps}")
    
    has_non_numeric = False
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            has_non_numeric = True
            break
            
    if has_non_numeric:
        logger.warning("Coercing non-numeric Beta-values to numeric (NaNs will be introduced)")
        
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    B = np.clip(df_numeric.values, eps, 1.0 - eps)
    M = np.log2(B / (1.0 - B))
    
    logger.info("Beta-to-M conversion complete.")

    return pd.DataFrame(M, index=df_numeric.index, columns=df_numeric.columns)