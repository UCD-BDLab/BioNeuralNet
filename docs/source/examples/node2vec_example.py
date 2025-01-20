import pandas as pd
from bioneuralnet.external_tools import node2vec


def main():
    try:
        print("Starting Node2Vec Embedding Workflow...")

        adjacency_matrix = pd.DataFrame(
            {
                "GeneA": [1.0, 1.0, 0.0, 0.0],
                "GeneB": [1.0, 1.0, 1.0, 0.0],
                "GeneC": [0.0, 1.0, 1.0, 1.0],
                "GeneD": [0.0, 0.0, 1.0, 1.0],
            },
            index=["GeneA", "GeneB", "GeneC", "GeneD"],
        )

        node2vec_embedding = node2vec(
            adjacency_matrix=adjacency_matrix,
            embedding_dim=64,
            walk_length=30,
            num_walks=200,
            window_size=10,
            workers=4,
            seed=42,
        )

        embeddings = node2vec_embedding.run()

        print("\nNode Embeddings:")
        print(embeddings)

        output_file = "output/embeddings.csv"
        embeddings.to_csv(output_file)

        print("\nNode2Vec Embedding Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e


if __name__ == "__main__":
    main()
