# /m3_network_embedding/config/node2vec.py

import logging
import pandas as pd
import os
from node2vec import Node2Vec
import networkx as nx

def run_node2vec(cluster_file, config, output_dir):
    """
    Perform node2vec embedding on the network data.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running node2vec Embedding on {cluster_file}")

    try:
        # Load cluster data
        cluster_df = pd.read_csv(cluster_file, index_col=0)

        # Create a graph from the cluster adjacency matrix
        G = nx.from_pandas_adjacency(cluster_df)

        # Access node2vec parameters
        node2vec_params = config['network_embedding']['node2vec']

        # Initialize node2vec model
        node2vec = Node2Vec(
            G,
            dimensions=node2vec_params['embedding_dim'],
            walk_length=node2vec_params['walk_length'],
            num_walks=node2vec_params['num_walks'],
            workers=4,
            seed=42
        )

        # Fit the model
        model = node2vec.fit(window=node2vec_params['window_size'], min_count=1, batch_words=4)

        # Save embeddings
        embeddings_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(cluster_file))[0]}_embeddings.csv")
        embeddings_df = pd.DataFrame({
            'node': model.wv.index_to_key,
            'embedding': model.wv.vectors.tolist()
        })
        embeddings_df.to_csv(embeddings_file, index=False)
        logger.info(f"node2vec embeddings saved to {embeddings_file}")

    except Exception as e:
        logger.error(f"Error in node2vec Embedding: {e}")
        raise
