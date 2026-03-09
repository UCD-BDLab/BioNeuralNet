import pandas as pd
import numpy as np
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)

def variance_summary(df: pd.DataFrame, low_var_threshold: Optional[float] = None) -> dict:
    """Computes key summary statistics for the feature (column) variances within an omics DataFrame.

    This is useful for assessing feature distribution and identifying low-variance features prior to modeling.

    Args:

        df (pd.DataFrame): The input omics DataFrame (samples as rows, features as columns).
        low_var_threshold (Optional[float]): A threshold used to count features falling below this variance level.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and standard deviation of the column variances. 
              If a threshold is provided, it also includes 'num_low_variance_features'.

    """
    variances = df.var()
    summary = {
        "variance_mean": float(variances.mean()),
        "variance_median": float(variances.median()),
        "variance_min": float(variances.min()),
        "variance_max": float(variances.max()),
        "variance_std": float(variances.std())
    }
    if low_var_threshold is not None:
        summary["num_low_variance_features"] = int((variances < low_var_threshold).sum())

    return summary

def zero_fraction_summary(df: pd.DataFrame, high_zero_threshold: Optional[float] = None) -> dict:
    """Computes statistics on the fraction of zero values present in each feature (column).

    This helps identify feature sparsity, which is common in omics data (e.g., RNA-seq FPKM).

    Args:

        df (pd.DataFrame): The input omics DataFrame.
        high_zero_threshold (Optional[float]): A threshold used to count features whose zero-fraction exceeds this value.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and standard deviation of the zero fractions. 
              If a threshold is provided, it includes 'num_high_zero_features'.

    """
    zero_fraction = (df == 0).sum(axis=0) / df.shape[0]
    summary = {
        "zero_fraction_mean": float(zero_fraction.mean()),
        "zero_fraction_median": float(zero_fraction.median()),
        "zero_fraction_min": float(zero_fraction.min()),
        "zero_fraction_max": float(zero_fraction.max()),
        "zero_fraction_std": float(zero_fraction.std())
    }
    if high_zero_threshold is not None:
        summary["num_high_zero_features"] = int((zero_fraction > high_zero_threshold).sum())

    return summary

def expression_summary(df: pd.DataFrame) -> dict:
    """Computes summary statistics for the mean expression (average value) of all features.

    Provides insight into the overall magnitude and central tendency of the data values.

    Args:

        df (pd.DataFrame): The input omics DataFrame.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and standard deviation of the feature means.

    """
    mean_expression = df.mean()
    summary = {
        "expression_mean": float(mean_expression.mean()),
        "expression_median": float(mean_expression.median()),
        "expression_min": float(mean_expression.min()),
        "expression_max": float(mean_expression.max()),
        "expression_std": float(mean_expression.std())
    }
    return summary

def correlation_summary(df: pd.DataFrame) -> dict:
    """Computes summary statistics on the maximum pairwise (absolute) correlation observed for each feature.

    This helps identify features that are highly redundant or collinear.

    Args:

        df (pd.DataFrame): The input omics DataFrame.

    Returns:

        dict: A dictionary containing the mean, median, min, max, and std of the max absolute correlations.

    """
    corr_matrix = df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = corr_matrix.max()

    summary = {
        "max_corr_mean": float(max_corr.mean()),
        "max_corr_median": float(max_corr.median()),
        "max_corr_min": float(max_corr.min()),
        "max_corr_max": float(max_corr.max()),
        "max_corr_std": float(max_corr.std())
    }
    return summary

def explore_data_stats(df: pd.DataFrame, name: str = "Data") -> None:
    """Prints a comprehensive set of key statistics for an omics DataFrame.

    Combines variance, zero fraction, expression, and correlation summaries for rapid data quality assessment.

    Args:

        df (pd.DataFrame): The input omics DataFrame.
        name (str): A descriptive name for the dataset (e.g., 'X_rna_final') for clear output labeling.

    Returns:

        None: Prints the statistics directly to the console.

    """
    logger.info(f"Statistics for {name}:")
    var_stats = variance_summary(df, low_var_threshold=1e-4)
    logger.info(f"Variance Summary: {var_stats}")

    zero_stats = zero_fraction_summary(df, high_zero_threshold=0.50)
    logger.info(f"Zero Fraction Summary: {zero_stats}")

    expr_stats = expression_summary(df)
    logger.info(f"Expression Summary: {expr_stats}")

    try:
        corr_stats = correlation_summary(df)
        logger.info(f"Correlation Summary: {corr_stats}")
    except Exception as e:
        logger.info(f"Correlation Summary: Could not compute due to: {e}")

    logger.info("\n")