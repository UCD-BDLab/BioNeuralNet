import logging
import pandas as pd
import networkx as nx
import os

def run_pagerank(network_file, config, output_dir):
    """
    Perform PageRank-based Clustering on the network data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running PageRank Clustering")

    try:
        # Load network data
        network_df = pd.read_csv(network_file)
        # Create a graph from the network data
        G = nx.from_pandas_edgelist(network_df, 'source', 'target', ['weight'])  # Adjust columns as necessary

        # Compute PageRank scores
        pagerank_scores = nx.pagerank(G, alpha=config['PageRank']['damping_factor'], max_iter=config['PageRank']['max_iter'], tol=config['PageRank']['tol'])

        # Convert to DataFrame
        pagerank_df = pd.DataFrame(list(pagerank_scores.items()), columns=['node', 'pagerank'])

        # Save PageRank scores
        pagerank_file = os.path.join(output_dir, "pagerank_scores.csv")
        pagerank_df.to_csv(pagerank_file, index=False)
        logger.info(f"PageRank scores saved to {pagerank_file}")

        # Optionally, perform clustering based on PageRank scores
        # Example: Simple threshold-based clustering
        threshold = pagerank_df['pagerank'].median()
        pagerank_df['cluster'] = pagerank_df['pagerank'].apply(lambda x: 'High' if x >= threshold else 'Low')

        # Save cluster labels
        cluster_labels_file = os.path.join(output_dir, "cluster_labels.csv")
        pagerank_df.to_csv(cluster_labels_file, index=False)
        logger.info(f"Cluster labels saved to {cluster_labels_file}")

    except Exception as e:
        logger.error(f"Error in PageRank Clustering: {e}")
        raise
