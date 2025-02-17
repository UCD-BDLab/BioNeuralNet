import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from bioneuralnet.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

def compute_correlation(predictions: pd.Series, targets: pd.Series) -> float:
    """
    Compute the Pearson correlation coefficient between predictions and targets.
    """
    logger.info("Computing Pearson correlation coefficient.")
    if predictions.empty or targets.empty:
        logger.error("Predictions and targets must not be empty.")
        raise ValueError("Predictions and targets must not be empty.")
    if len(predictions) != len(targets):
        logger.error("Predictions and targets must have the same length.")
        raise ValueError("Predictions and targets must have the same length.")
    correlation = predictions.corr(targets)
    logger.info(f"Pearson correlation coefficient: {correlation}")
    return correlation if not pd.isna(correlation) else 0.0


def compute_cluster_correlation_from_df(cluster_df: pd.DataFrame, pheno: pd.DataFrame) -> tuple:
    """
    Compute the Pearson correlation coefficient between PC1 of a cluster and phenotype.
    
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

def convert_louvain_to_adjacency(louvain_cluster: pd.DataFrame) -> pd.DataFrame:
    adjacency_matrix = louvain_cluster.corr(method="pearson")
    np.fill_diagonal(adjacency_matrix.values, 0)
    adjacency_matrix = adjacency_matrix.fillna(0)
    return adjacency_matrix
