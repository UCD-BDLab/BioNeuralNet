import pandas as pd
from bioneuralnet.network_embedding import GnnEmbedding

def main():
    omics_files = [pd.read_csv('input/proteins.csv', index_col=0),
                   pd.read_csv('input/metabolites.csv', index_col=0)]
    phenotype_df = pd.read_csv('input/phenotype_data.csv', index_col=0)
    clinical_data_df = pd.read_csv('input/clinical_data.csv', index_col=0)
    adjacency_matrix = pd.read_csv('input/adjacency_matrix.csv', index_col=0)

    gnn_embed = GnnEmbedding(
        omics_list=omics_files,
        phenotype_df=phenotype_df,
        clinical_data_df=clinical_data_df,
        adjacency_matrix=adjacency_matrix,
        model_type='GCN'
    )

    embeddings_dict = gnn_embed.run()
    embeddings = embeddings_dict['graph']
    print("GNN Embeddings generated successfully.")
    print(embeddings)

if __name__ == "__main__":
    main()
