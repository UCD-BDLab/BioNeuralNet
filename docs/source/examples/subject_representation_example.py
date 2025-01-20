import pandas as pd
from bioneuralnet.subject_representation import GraphEmbedding


def main():
    phenotype_df = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "Phenotype": ["Control", "Treatment", "Control", "Treatment"],
        },
        index_col=0,
    )

    omics_df1 = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "GeneA": [1.2, 2.3, 3.1, 4.0],
            "GeneB": [2.1, 3.4, 1.2, 3.3],
            "GeneC": [3.3, 1.5, 2.2, 4.1],
        },
        index_col=0,
    )

    omics_df2 = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "GeneD": [4.2, 5.3, 6.1, 7.0],
            "GeneE": [5.1, 6.4, 4.2, 6.3],
            "GeneF": [6.3, 4.5, 5.2, 7.1],
        },
        index_col=0,
    )

    # Example Clinical DataFrame
    clinical_data_df = pd.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "Age": [30, 40, 50, 60],
            "Sex": ["Male", "Female", "Female", "Male"],
            "BMI": [25.0, 28.1, 30.2, 24.5],
        },
        index_col=0,
    )

    adjacency_matrix = pd.DataFrame(
        {
            "GeneA": [1.0, 0.8, 0.3, 0.0],
            "GeneB": [0.8, 1.0, 0.4, 0.0],
            "GeneC": [0.3, 0.4, 1.0, 0.7],
            "GeneD": [0.0, 0.0, 0.7, 1.0],
        },
        index=["GeneA", "GeneB", "GeneC", "GeneD"],
        index_col=0,
    )

    # You can also load the data from files directly

    # omics_files = [pd.read_csv('input/proteins.csv', index_col=0),
    #                pd.read_csv('input/metabolites.csv', index_col=0)]
    # phenotype_df = pd.read_csv('input/phenotype_data.csv', index_col=0)
    # clinical_data_df = pd.read_csv('input/clinical_data.csv', index_col=0)
    # adjacency_matrix = pd.read_csv('input/adjacency_matrix.csv', index_col=0)

    graph_embed = GraphEmbedding(
        adjacency_matrix=adjacency_matrix,
        omics_list=[omics_df1, omics_df2],
        phenotype_df=phenotype_df,
        clinical_data_df=clinical_data_df,
        embeddings="GNNs",
    )

    enhanced_omics_data = graph_embed.run()

    enhanced_omics_data.to_csv("output/enhanced_omics_data.csv")
    print("Graph embedding workflow completed successfully.")


if __name__ == "__main__":
    main()
