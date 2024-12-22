import pandas as pd
from bioneuralnet.network_embedding import Node2VecEmbedding


def main():
    try:
        print("Starting Node2Vec Embedding Workflow...")

        adjacency_matrix = pd.DataFrame({
            'GeneA': [1.0, 1.0, 0.0, 0.0],
            'GeneB': [1.0, 1.0, 1.0, 0.0],
            'GeneC': [0.0, 1.0, 1.0, 1.0],
            'GeneD': [0.0, 0.0, 1.0, 1.0]
        }, index=['GeneA', 'GeneB', 'GeneC', 'GeneD'])

        node2vec = Node2VecEmbedding(
            adjacency_matrix=adjacency_matrix,
            embedding_dim=64,      
            walk_length=30,        
            num_walks=200,        
            window_size=10,        
            workers=4,             
            seed=42,                
        )

        embeddings = node2vec.run()

        print("\nNode Embeddings:")
        print(embeddings)

        # save_path = 'node_embeddings.csv'
        # node2vec.save_embeddings(save_path)
        # print(f"\nEmbeddings saved to {save_path}")

        # We have a built in function to save the embeddings to a csv file
        # But we can also save the embeddings to a csv file using the following code
        output_file = 'output/embeddings.csv'
        embeddings.to_csv(output_file)

        print("\nNode2Vec Embedding Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e


if __name__ == "__main__":
    main()
