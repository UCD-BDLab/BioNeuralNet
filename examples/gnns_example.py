import pandas as pd
from bioneuralnet.network_embedding.gnns import GNNEmbedding

def main():
    # Paths to input files
    omics_files = ['input/proteins.csv', 'input/metabolites.csv']
    phenotype_file = 'input/phenotype_data.csv'
    clinical_data_file = 'input/clinical_data.csv' 
    adjacency_matrix_file = 'input/adjacency_matrix.csv'

    # Load adjacency matrix
    adjacency_matrix = pd.read_csv(adjacency_matrix_file, index_col=0)

    # Initialize GNNEmbedding
    gnn_embedding = GNNEmbedding(
        omics_list=omics_files,
        phenotype_file=phenotype_file,
        clinical_data_file=clinical_data_file,
        adjacency_matrix=adjacency_matrix,
        model_type='GCN', 
        gnn_hidden_dim=64,
        gnn_layer_num=2,
        dropout=True
    )

    # Run GNN embedding to generate embeddings
    embeddings_dict = gnn_embedding.run()

    # Access embeddings
    embeddings_tensor = embeddings_dict['graph']
    embeddings_df = pd.DataFrame(
        embeddings_tensor.numpy(),
        index=adjacency_matrix.index,
        columns=[f"dim_{i}" for i in range(embeddings_tensor.shape[1])]
    )

    print("Embeddings generated successfully.")
    print("Sample embeddings:")
    print(embeddings_df.head())

if __name__ == "__main__":
    main()
