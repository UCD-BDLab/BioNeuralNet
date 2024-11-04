import logging
import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering

def run_hierarchical(network_file, config, output_dir):
    """
    Perform Hierarchical Clustering on the network data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running Hierarchical Clustering")

    try:
        # Load network data
        network_df = pd.read_csv(network_file, index_col=0)
        feature_matrix = network_df.values

        # Initialize the clustering model
        model = AgglomerativeClustering(
            n_clusters=config['clustering']['Hierarchical']['n_clusters'],
            linkage=config['clustering']['Hierarchical']['linkage'],
            metric=config['clustering']['Hierarchical']['affinity']
        )

        # Fit the model
        labels = model.fit_predict(feature_matrix)

        # Save cluster labels
        cluster_labels_file = os.path.join(output_dir, "cluster_labels.csv")
        cluster_labels_df = pd.DataFrame({
            'node': network_df.index,
            'cluster': labels
        })
        cluster_labels_df.to_csv(cluster_labels_file, index=False)
        logger.info(f"Cluster labels saved to {cluster_labels_file}")

        n_clusters = config['clustering']['Hierarchical']['n_clusters']

        for i in range(n_clusters):
            cluster_nodes = network_df.index[labels == i]
            cluster_file = os.path.join(output_dir, f"cluster_{i+1}.csv")
            cluster_data = network_df.loc[cluster_nodes, cluster_nodes]
            cluster_data.to_csv(cluster_file)
            logger.info(f"Cluster {i+1} data saved to {cluster_file}")

    except Exception as e:
        logger.error(f"Error in Hierarchical Clustering: {e}")
        raise
