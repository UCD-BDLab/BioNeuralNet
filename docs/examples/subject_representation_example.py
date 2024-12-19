import pandas as pd
from bioneuralnet.subject_representation import GraphEmbedding

def main():
    omics_files = [pd.read_csv('input/proteins.csv', index_col=0),
                   pd.read_csv('input/metabolites.csv', index_col=0)]
    phenotype_df = pd.read_csv('input/phenotype_data.csv', index_col=0)
    clinical_data_df = pd.read_csv('input/clinical_data.csv', index_col=0)
    adjacency_matrix = pd.read_csv('input/adjacency_matrix.csv', index_col=0)

    graph_embed = GraphEmbedding(
        adjacency_matrix=adjacency_matrix,
        omics_list=omics_files,
        phenotype_df=phenotype_df,
        clinical_data_df=clinical_data_df,
        embedding_method='GNNs'
    )

    # Run the graph embedding process
    enhanced_omics_data = graph_embed.run()

    enhanced_omics_data.to_csv('output/enhanced_omics_data.csv')
    print("Graph embedding workflow completed successfully.")

if __name__ == "__main__":
    main()
