"""
Example 1: Sparse Multiple Canonical Correlation Network (SmCCNet) Workflow with Graph Neural Network (GNN) Embeddings
======================================================================================================================

Steps:
1. Generate an adjacency matrix using SmCCNet based on multi-omics and phenotype data.
2. Generate GNN node embeddings (GNNEmbedding) using the adjacency, omics, phenotype, and clinical data.
3. Pass these precomputed embeddings into GraphEmbedding to integrate into omics data.
"""

import pandas as pd
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.network_embedding import GNNEmbedding
from bioneuralnet.subject_representation import GraphEmbedding


def run_smccnet_workflow(
    omics_data: pd.DataFrame, phenotype_data: pd.DataFrame, clinical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Executes the SmCCNet-based workflow for generating enhanced omics data.

    1) Instantiates SmCCNet, GNNEmbedding, and GraphEmbedding.
    2) Generates an <ulti-Omics Network (adjacency matrix) via SmCCNet.
    3) Runs GNNEmbedding to produce node embeddings.
    4) Integrates the embeddings into omics data using GraphEmbedding.

    Args:
        omics_data (pd.DataFrame): Omics features (e.g., proteins, metabolites).
        phenotype_data (pd.Series): Phenotype info (numeric/binary).
        clinical_data (pd.DataFrame): Clinical info.

    Returns:
        pd.DataFrame: Enhanced omics data integrated with GNN embeddings.
    """
    try:
        smccnet_instance = SmCCNet(
            phenotype_df=phenotype_data,
            omics_dfs=omics_data,
            data_types=["protein", "metabolite"],
            kfold=5,
            summarization="PCA",
            seed=732,
        )
        adjacency_matrix = smccnet_instance.run()
        print("Adjacency matrix generated using SmCCNet.")

        gnn_embedding = GNNEmbedding(
            adjacency_matrix=adjacency_matrix,
            omics_data=omics_data,
            phenotype_data=phenotype_data,
            clinical_data=clinical_data,
            model_type="GCN",
            hidden_dim=64,
            layer_num=2,
            dropout=True,
            num_epochs=50,
            lr=1e-3,
            weight_decay=1e-4,
        )
        embeddings_dict = gnn_embedding.run()
        embeddings_tensor = embeddings_dict["graph"]

        embeddings_df = pd.DataFrame(
            embeddings_tensor.numpy(), index=adjacency_matrix.index
        )
        print("GNN embeddings generated. Shape:", embeddings_df.shape)

        graph_embedding = GraphEmbedding(
            adjacency_matrix=adjacency_matrix,
            omics_data=omics_data,
            phenotype_data=phenotype_data,
            clinical_data=clinical_data,
            embeddings=embeddings_df,
        )
        enhanced_omics_data = graph_embedding.run()
        print("Embeddings integrated into omics data.")

        return enhanced_omics_data

    except Exception as e:
        print(f"An error occurred during the SmCCNet workflow: {e}")
        raise e


if __name__ == "__main__":
    try:
        print("Starting SmCCNet + GNNEmbedding + GraphEmbedding Workflow...")

        omics_data = pd.DataFrame(
            {
                "protein_feature1": [0.1, 0.2],
                "protein_feature2": [0.3, 0.4],
                "metabolite_feature1": [0.5, 0.6],
                "metabolite_feature2": [0.7, 0.8],
            },
            index=["Sample1", "Sample2"],
        )

        phenotype_data = pd.Series([1, 0], index=["Sample1", "Sample2"])

        clinical_data = pd.DataFrame(
            {"clinical_feature1": [5, 3], "clinical_feature2": [7, 2]},
            index=["Sample1", "Sample2"],
        )
        enhanced_omics = run_smccnet_workflow(
            omics_data=omics_data,
            phenotype_data=phenotype_data,
            clinical_data=clinical_data,
        )

        print("\nEnhanced Omics Data:")
        print(enhanced_omics)

        print("SmCCNet workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e
