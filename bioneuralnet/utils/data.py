import pandas as pd
import numpy as np
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)

def variance_summary(df: pd.DataFrame, var_threshold: Optional[float] = None) -> dict:
    """Computes key summary statistics for the feature (column) variances within an omics DataFrame.

    This is useful for assessing feature distribution and identifying low-variance features prior to modeling.

    Args:
        df (pd.DataFrame): The input omics DataFrame (samples as rows, features as columns).
        var_threshold (Optional[float]): A threshold used to count features falling below this variance level.

    Returns:
        dict: A dictionary containing the mean, median, min, max, and standard deviation of the column variances.
              If a threshold is provided, it also includes 'Number Of Low Variance Features'.
    """
    variances = df.var()
    summary = {
        "Variance Mean": float(variances.mean()),
        "Variance Median": float(variances.median()),
        "Variance Min": float(variances.min()),
        "Variance Max": float(variances.max()),
        "Variance Std": float(variances.std())
    }
    if var_threshold is not None:
        summary["Number Of Low Variance Features"] = int((variances < var_threshold).sum())

    return summary

def zero_summary(df: pd.DataFrame, zero_threshold: Optional[float] = None) -> dict:
    """Computes statistics on the fraction of zero values present in each feature (column).

    This helps identify feature sparsity, which is common in omics data (e.g., RNA-seq FPKM).

    Args:
        df (pd.DataFrame): The input omics DataFrame.
        zero_threshold (Optional[float]): A threshold used to count features whose zero-fraction exceeds this value.

    Returns:
        dict: A dictionary containing the mean, median, min, max, and standard deviation of the zero fractions.
              If a threshold is provided, it includes 'Number Of High Zero Features'.
    """
    zero_fraction = (df == 0).sum(axis=0) / df.shape[0]
    summary = {
        "Zero Mean": float(zero_fraction.mean()),
        "Zero Median": float(zero_fraction.median()),
        "Zero Min": float(zero_fraction.min()),
        "Zero Max": float(zero_fraction.max()),
        "Zero Std": float(zero_fraction.std())
    }
    if zero_threshold is not None:
        summary["Number Of High Zero Features"] = int((zero_fraction > zero_threshold).sum())

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
        "Expression Mean": float(mean_expression.mean()),
        "Expression Median": float(mean_expression.median()),
        "Expression Min": float(mean_expression.min()),
        "Expression Max": float(mean_expression.max()),
        "Expression Std": float(mean_expression.std())
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
    corr_matrix = corr_matrix.copy()
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = corr_matrix.max()

    summary = {
        "Max Corr Mean": float(max_corr.mean()),
        "Max Corr Median": float(max_corr.median()),
        "Max Corr Min": float(max_corr.min()),
        "Max Corr Max": float(max_corr.max()),
        "Max Corr Std": float(max_corr.std())
    }
    return summary

def nan_summary(df: pd.DataFrame, name: str = "Dataset", missing_threshold: float = 20.0) -> float:
    """Logs a report on the missing data (NaNs) in the DataFrame.

    Args:
        df (pd.DataFrame): The input omics DataFrame.
        name (str): A descriptive name for the dataset for clear output labeling.
        missing_threshold (float): Percentage threshold (0-100) to trigger a warning for highly missing data.

    Returns:
        float: The global percentage of missing values (NaNs) in the DataFrame.
    """
    total_elements = df.size
    total_nans = df.isna().sum().sum()
    pct_missing = (total_nans / total_elements) * 100

    logger.info(f"=== {name} NaN Report ===")
    logger.info(f"Shape: {df.shape} (Samples: {df.shape[0]}, Features: {df.shape[1]})")
    logger.info(f"Global NaN: {pct_missing:.2f}%\n")

    if total_nans > 0:
        feature_nan_pct = (df.isna().sum(axis=0) / df.shape[0]) * 100
        sample_nan_pct = (df.isna().sum(axis=1) / df.shape[1]) * 100

        logger.info("Top 5 FEATURES with most missing data:")
        logger.info("\n" + feature_nan_pct.sort_values(ascending=False).head(5).to_string(float_format="{:.2f}%".format))

        logger.info("\nTop 5 SAMPLES with most missing data:")
        logger.info("\n" + sample_nan_pct.sort_values(ascending=False).head(5).to_string(float_format="{:.2f}%".format))

        high_missing_features = (feature_nan_pct > missing_threshold).sum()
        high_missing_samples = (sample_nan_pct > missing_threshold).sum()

        if high_missing_features > 0:
            logger.warning(f"{high_missing_features} features are missing in >{missing_threshold}% of samples.")
        if high_missing_samples > 0:
            logger.warning(f"{high_missing_samples} samples are missing >{missing_threshold}% of their features.")

    logger.info("-" * 50)
    return pct_missing


def sparse_filter(df: pd.DataFrame, missing_fraction: float = 0.20) -> pd.DataFrame:
    """Drops features (columns) and then samples (rows) that exceed the maximum missing data fraction.

    Args:
        df (pd.DataFrame): The input omics DataFrame.
        missing_fraction (float): The maximum allowed fraction of missing values (0.0 to 1.0).

    Returns:
        pd.DataFrame: The filtered DataFrame with highly missing features and samples removed.
    """
    min_valid_samples = int(df.shape[0] * (1 - missing_fraction))
    df_filtered = df.dropna(axis=1, thresh=min_valid_samples)

    min_valid_features = int(df_filtered.shape[1] * (1 - missing_fraction))
    return df_filtered.dropna(axis=0, thresh=min_valid_features)

def data_stats(df: pd.DataFrame, name: str = "Data", compute_correlation: bool = False) -> None:
    """Prints a comprehensive set of key statistics for an omics DataFrame.

    Combines variance, zero fraction, expression, correlation, and missingness summaries
    for rapid data quality assessment. Recommends data cleaning steps if high missingness is found.

    Args:
        df (pd.DataFrame): The input omics DataFrame.
        name (str): A descriptive name for the dataset (e.g., 'X_rna_final') for clear output labeling.
        compute_correlation (bool): Whether to compute pairwise correlations. Defaults to False.

    Returns:
        None: Logs the statistics directly to the console.
    """
    logger.info(f"=== {name} Statistics Overview ===")

    var_stats = variance_summary(df, var_threshold=1e-4)
    logger.info("--- Variance Summary ---")
    for key, value in var_stats.items():
        clean_val = f"{value:.4f}" if isinstance(value, float) else str(value)
        logger.info(f"{key:<32}: {clean_val}")
    logger.info("")

    zero_stats = zero_summary(df, zero_threshold=0.50)
    logger.info("--- Zero Summary ---")
    for key, value in zero_stats.items():
        clean_val = f"{value:.4f}" if isinstance(value, float) else str(value)
        logger.info(f"{key:<32}: {clean_val}")
    logger.info("")

    expr_stats = expression_summary(df)
    logger.info("--- Expression Summary ---")
    for key, value in expr_stats.items():
        clean_val = f"{value:.4f}" if isinstance(value, float) else str(value)
        logger.info(f"{key:<32}: {clean_val}")
    logger.info("")

    if compute_correlation:
        try:
            corr_stats = correlation_summary(df)
            logger.info("--- Correlation Summary ---")
            for key, value in corr_stats.items():
                clean_val = f"{value:.4f}" if isinstance(value, float) else str(value)
                logger.info(f"{key:<32}: {clean_val}")
            logger.info("")
        except Exception as e:
            logger.info("--- Correlation Summary ---")
            logger.info(f"Could not compute due to: {e}\n")
    else:
        logger.info("--- Correlation Summary ---")
        logger.info(f"{'Skipped':<32}: (compute_correlation=False)\n")

    pct_missing = nan_summary(df, name=name)

    logger.info(f"--- {name} Recommendations ---")

    # 1. Missingness Check
    if pct_missing > 30.0:
        logger.warning(
            f"SPARSITY: {pct_missing:.2f}% missing data. "
            f"Consider running `df = sparse_filter(df, missing_fraction=0.30)`."
        )

    # 2. Beta Value Check (Bounded between 0 and 1)
    expr_min = expr_stats["Expression Min"]
    expr_max = expr_stats["Expression Max"]

    if expr_min >= 0.0 and expr_max <= 1.0:
        logger.warning(
            "NORMALIZATION: Values are strictly bounded between 0 and 1. "
            "If these are Methylation Beta values, highly consider applying `m_transform(df)` "
            "to convert them to M-values for neural network stability."
        )
    # 3. Raw Counts Check (High exact zeros)
    elif zero_stats.get("Number Of High Zero Features", 0) > 0:
        logger.warning(
            "NORMALIZATION: High number of exact zeros detected. If these are raw RNA/miRNA counts, "
            "consider a log2 transformation prior to modeling."
        )
    else:
        logger.info("NORMALIZATION: Data distribution looks unbounded with low exact zeros. "
                    "Appears properly transformed.")

    logger.info("=" * 50 + "\n")
