import pandas as pd
import numpy as np
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)

def variance_summary(df: pd.DataFrame, low_var_threshold: Optional[float] = None) -> dict:
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

def zero_fraction_summary(df: pd.DataFrame, high_zero_threshold: Optional[float] = None) -> dict:
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
