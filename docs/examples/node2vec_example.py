"""
Example: Node2VecEmbedding Usage
===============================
This script demonstrates how to use the refactored Node2VecEmbedding class with in-memory data structures.
It generates Node2Vec-based embeddings for a provided graph represented as a pandas DataFrame.
"""

import pandas as pd
from bioneuralnet.network_embedding import Node2VecEmbedding


def main():
    try:
        print("Starting Node2Vec Embedding Workflow...")

        # Example Adjacency Matrix Data
        # Replace this with your actual adjacency matrix DataFrame
        adjacency_matrix = pd.DataFrame({
            'GeneA': [1.0, 1.0, 0.0, 0.0],
            'GeneB': [1.0, 1.0, 1.0, 0.0],
            'GeneC': [0.0, 1.0, 1.0, 1.0],
            'GeneD': [0.0, 0.0, 1.0, 1.0]
        }, index=['GeneA', 'GeneB', 'GeneC', 'GeneD'])

        # Initialize Node2VecEmbedding Instance
        node2vec = Node2VecEmbedding(
            adjacency_matrix=adjacency_matrix,
            embedding_dim=64,      # Dimension of the embeddings
            walk_length=30,        # Length of each walk
            num_walks=200,         # Number of walks per node
            window_size=10,        # Window size for Word2Vec
            workers=4,             # Number of worker threads
            seed=42                # Random seed for reproducibility
        )

        # Run Node2Vec Embedding
        embeddings = node2vec.run()

        # Display Embeddings
        print("\nNode Embeddings:")
        print(embeddings)

        # Optionally, save embeddings to a CSV file
        save_path = 'node_embeddings.csv'
        node2vec.save_embeddings(save_path)
        print(f"\nEmbeddings saved to {save_path}")

        print("\nNode2Vec Embedding Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e


if __name__ == "__main__":
    main()
