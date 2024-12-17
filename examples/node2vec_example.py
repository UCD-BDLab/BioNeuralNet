from bioneuralnet.network_embedding import Node2VecEmbedding

def main():
    # Initialize Node2VecEmbedding parameters
    node2vec_embedding = Node2VecEmbedding(
        input_dir='input/graphs/',    
        embedding_dim=128,            
        walk_length=80,              
        num_walks=10,                 
        window_size=10,              
        workers=4,                   
        seed=42,                      
        output_dir=None               
    )

    # Run Node2Vec to generate embeddings
    embeddings = node2vec_embedding.run()

    # embeddings for a specific graph
    graph_name = 'global_network' 
    if graph_name in embeddings:
        embeddings_df = embeddings[graph_name]
        print(f"Embeddings for '{graph_name}' generated successfully.")
    else:
        print(f"No embeddings found for '{graph_name}'.")

if __name__ == "__main__":
    main()
