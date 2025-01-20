import pandas as pd
from bioneuralnet.network_embedding import GNNEmbedding


def main():
    phenotype_df = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "Phenotype": ["Control", "Treatment", "Control", "Treatment"],
        }
    )

    omics_df1 = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "GeneA": [1.2, 2.3, 3.1, 4.0],
            "GeneB": [2.1, 3.4, 1.2, 3.3],
            "GeneC": [3.3, 1.5, 2.2, 4.1],
        }
    )

    omics_df2 = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "GeneD": [4.2, 5.3, 6.1, 7.0],
            "GeneE": [5.1, 6.4, 4.2, 6.3],
            "GeneF": [6.3, 4.5, 5.2, 7.1],
        }
    )

    clinical_data_df = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "Age": [30, 40, 50, 60],
            "Sex": ["Male", "Female", "Female", "Male"],
            "BMI": [25.0, 28.1, 30.2, 24.5],
        }
    )

    adjacency_matrix = pd.DataFrame(
        {
            "GeneA": [1.0, 0.8, 0.3, 0.0],
            "GeneB": [0.8, 1.0, 0.4, 0.0],
            "GeneC": [0.3, 0.4, 1.0, 0.7],
            "GeneD": [0.0, 0.0, 0.7, 1.0],
        },
        index=["GeneA", "GeneB", "GeneC", "GeneD"],
    )

    omics_data = pd.concat([omics_df1, omics_df2], axis=1)

    gnn_embed = GNNEmbedding(
        adjacency_matrix=adjacency_matrix,
        omics_data=omics_data,
        phenotype_df=phenotype_df.set_index("SampleID"),
        clinical_data_df=clinical_data_df.set_index("SampleID"),
        model_type="GCN",
    )

    print("Generating GNN embeddings...")
    embeddings_dict = gnn_embed.run()
    embeddings = embeddings_dict["graph"]

    print("GNN Embeddings generated successfully.")
    print(embeddings.head())


if __name__ == "__main__":
    main()
