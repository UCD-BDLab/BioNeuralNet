import pandas as pd
from bioneuralnet.external_tools import FeatureSelector
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

    gnn_embed = GNNEmbedding(
        omics_list=[omics_df1.set_index("SampleID"), omics_df2.set_index("SampleID")],
        phenotype_df=phenotype_df.set_index("SampleID"),
        clinical_data_df=clinical_data_df.set_index("SampleID"),
        adjacency_matrix=adjacency_matrix,
        model_type="GCN",
    )
    gnn_embeddings = gnn_embed.run()
    print("\nGNN Embeddings:")
    print(gnn_embeddings["graph"].head())

    combined_omics_data = pd.merge(omics_df1, omics_df2, on="SampleID")
    phenotype_series = phenotype_df.set_index("SampleID")["Phenotype"]

    feature_selector = FeatureSelector(
        enhanced_omics_data=combined_omics_data.set_index("SampleID"),
        phenotype_data=phenotype_series,
        num_features=20,
        selection_method="lasso",
    )
    selected_features = feature_selector.run_feature_selection()
    print("\nSelected Multi-Omics Features:")
    print(selected_features.head())


if __name__ == "__main__":
    main()
