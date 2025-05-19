import numpy as np
import pandas as pd
from typing import Tuple

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from bioneuralnet.utils.logger import get_logger

logger = get_logger(__name__)

def omics_correlation(omics: pd.DataFrame, pheno: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute the Pearson correlation between a group of omics data (reduced to one principal component)
    and a phenotype.

    Parameters:

        omics (pd.DataFrame): Omics data with rows as samples and columns as features.
        pheno (pd.DataFrame): Phenotype data. Expected to have a single column.

    Returns:

        Tuple[float, float]: Pearson correlation coefficient and p-value.
    """
    target = pheno.squeeze()

    logger.info("Computing Pearson correlation coefficient.")

    if omics.empty or target.empty:
        logger.error("Omics data and phenotype must not be empty.")
        raise ValueError("Omics data and phenotype must not be empty.")

    if omics.shape[0] != len(target):
        logger.error("Number of rows in omics data and phenotype must be the same.")
        raise ValueError("Omics data and phenotype must have the same length.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(omics)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(scaled).flatten()
    target_values = target.values

    corr, pvalue = pearsonr(pc1, target_values)

    return corr, pvalue


def cluster_correlation(cluster_df: pd.DataFrame, pheno: pd.DataFrame) -> tuple:
    """
    Compute the Pearson correlation coefficient between PC1 of a cluster and phenotype.

    Parameters:

        cluster_df: DataFrame representing a cluster of samples.
        pheno: DataFrame representing the phenotype.

    Returns:

        (cluster_size, correlation) or (size, None) if correlation fails.
    """
    cluster_size = cluster_df.shape[1]

    if cluster_size < 2:
        logger.info(f"Cluster with size {cluster_size} skipped (not enough features).")
        return (cluster_size, None)

    subset = cluster_df.fillna(0)

    if subset.var().sum() == 0:
        logger.warning("Cluster skipped: all features have zero variance.")
        return (cluster_size, None)

    try:
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(subset)
        pc1_series = pd.Series(pc1.flatten(), index=subset.index, name="PC1")

        pheno_series = pheno.iloc[:, 0]
        pc1_series, pheno_series = pc1_series.align(pheno_series, join="inner")

        if len(pc1_series) < 3:
            logger.warning("Not enough data points for Pearson correlation.")
            return (cluster_size, None)

        corr, _ = pearsonr(pc1_series, pheno_series)

    except Exception as e:
        logger.error(f"Error computing correlation: {e}")
        corr = None

    return (cluster_size, corr)

def louvain_to_adjacency(louvain_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a Louvain cluster to an adjacency matrix.

    Parameters:

        louvain_cluster: represents an induced subnetwork (from Louvain).

    Returns:

        pd.DataFrame: Adjacency matrix
    """
    adjacency_matrix = louvain_cluster.corr(method="pearson")
    np.fill_diagonal(adjacency_matrix.values, 0)
    adjacency_matrix = adjacency_matrix.fillna(0)

    return adjacency_matrix
