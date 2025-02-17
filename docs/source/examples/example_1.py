"""
Example 1: Sparse Multiple Canonical Correlation Network (SmCCNet) Workflow with Graph Neural Network (GNN) Embeddings
======================================================================================================================

Steps:
1. Load synthetic multi-omics and phenotype data using DatasetLoader.
2. Generate an adjacency matrix using SmCCNet based on multi-omics and phenotype data.
3. Generate GNN node embeddings (GNNEmbedding) using the adjacency, omics, phenotype, and clinical data.
4. Integrate the embeddings into omics data using GraphEmbedding to enhance feature representation.
"""

import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.network_embedding import GNNEmbedding
from bioneuralnet.subject_representation import GraphEmbedding

def run_smccnet_workflow() -> pd.DataFrame:
    """
    Executes the full SmCCNet-based workflow for generating enhanced omics data.

    Steps:
    1) Loads synthetic omics and phenotype data.
    2) Generates a Multi-Omics Network (adjacency matrix) using SmCCNet.
    3) Runs GNNEmbedding to produce node embeddings.
    4) Integrates embeddings into omics data using GraphEmbedding.

    Returns:
        pd.DataFrame: Enhanced omics data integrated with GNN embeddings.
    """
    try:
        # Load synthetic dataset
        print("Loading dataset...")
        loader = DatasetLoader("example1")
        omics1, omics2, phenotype, clinical = loader.load_data()

        # Display dataset dimensions
        print("Dataset Shapes:")
        print(f"Omics1: {omics1.shape}")
        print(f"Omics2: {omics2.shape}")
        print(f"Phenotype: {phenotype.shape}")
        print(f"Clinical: {clinical.shape}")

        # Merge omics data for processing
        merged_omics = pd.concat([omics1, omics2], axis=1)

        # Generate adjacency matrix using SmCCNet
        print("Running SmCCNet...")
        smccnet = SmCCNet(
            phenotype_df=phenotype,
            omics_dfs=[omics1, omics2],
            data_types=["genes", "proteins"],
            kfold=3,
            subSampNum=500,
        )
        global_network, smccnet_clusters = smccnet.run()
        print("Adjacency matrix generated using SmCCNet.")

        # Generate node embeddings using GNNEmbedding
        print("Running GNNEmbedding...")
        gnn_embedding = GNNEmbedding(
            adjacency_matrix=global_network,
            omics_data=merged_omics,
            phenotype_data=phenotype,
            clinical_data=clinical,
            tune=True,
        )
        gnn_embedding.fit()
        embeddings_output = gnn_embedding.embed(as_df=True)
        print(f"GNN embeddings generated. Shape: {embeddings_output.shape}")

        # Integrate embeddings into omics data using GraphEmbedding
        print("Running GraphEmbedding...")
        graph_embedding = GraphEmbedding(
            omics_data=merged_omics,
            embeddings=embeddings_output,
            phenotype_data=phenotype,
            tune=True,
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

        enhanced_omics = run_smccnet_workflow()

        print("\nEnhanced Omics Data:")
        print(enhanced_omics.head())

        print("\nSmCCNet workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e
