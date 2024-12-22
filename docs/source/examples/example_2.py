"""
Example 2: Weighted Gene Co-expression Network Analysis (WGCNA) Workflow with Graph Neural Network (GNN) Embeddings
=============================================================================================================

This script demonstrates a comprehensive workflow where we first generate a graph using Weighted Gene Co-expression Network Analysis
(WGCNA), and then use Graph Neural Network (GNN)-based embedding generation to create node
representations from the network. The process integrates the generated embeddings into subject-level omics data, enhancing
downstream analytical capabilities.

Steps:
1. Generate an adjacency matrix using WGCNA based on multi-omics and phenotype data.
2. Compute node features based on correlations.
3. Use a Graph Convolutional Network (GCN) to generate node embeddings.
4. Integrate the embeddings into the omics data for enhanced analysis.
"""

import pandas as pd
from bioneuralnet.graph_generation import WGCNA
from bioneuralnet.network_embedding import GNNEmbedding
from bioneuralnet.subject_representation import GraphEmbedding

def run_wgcna_workflow_with_gnn(omics_data: pd.DataFrame,
                                phenotype_data: pd.Series,
                                clinical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the WGCNA-based workflow for generating enhanced omics data.

    This function performs the following steps:
        1. Instantiates the WGCNA, GNNEmbedding, and GraphEmbedding components.
        2. Generates an adjacency matrix using WGCNA.
        3. Computes node features based on correlations.
        4. Generates embeddings using GNNEmbedding.
        5. Integrates embeddings into omics data to produce enhanced omics data.

    Args:
        omics_data (pd.DataFrame): DataFrame containing omics features (e.g., genes).
        phenotype_data (pd.Series): Series containing phenotype information.
        clinical_data (pd.DataFrame): DataFrame containing clinical data.

    Returns:
        pd.DataFrame: Enhanced omics data integrated with GNN embeddings.
    """
    try:
        wgcna_instance = WGCNA(
            phenotype_data=phenotype_data,
            omics_data=omics_data,
            data_types=['gene'], 
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
        )

        adjacency_matrix = wgcna_instance.run()
        print("Adjacency matrix generated using WGCNA.")

        node_features = omics_data[['gene_feature1', 'gene_feature2', 'gene_feature3', 'gene_feature4']] 

        gnn_embedding = GNNEmbedding(
            adjacency_matrix=adjacency_matrix,
            node_features=node_features,
            model_type='GCN', 
            gnn_hidden_dim=64,
            gnn_layer_num=2,
            dropout=True
        )
        embeddings_dict = gnn_embedding.run()
        embeddings_tensor = embeddings_dict['graph']
        embeddings_df = pd.DataFrame(embeddings_tensor.numpy(), index=node_features.index)
        print("GNN embeddings generated.")
        
        # Embeddings can also be saved to a file
        # output_file = 'output/embeddings.csv'
        # embeddings_df.to_csv(output_file)
        # print(f"Embeddings saved to {output_file}")

        # Step 4: Initialize and run GraphEmbedding
        graph_embedding = GraphEmbedding(
            adjacency_matrix=adjacency_matrix,
            omics_data=omics_data,
            phenotype_data=phenotype_data,
            clinical_data=clinical_data,
            embedding_method='GNNs'
        )
        enhanced_omics_data = graph_embedding.run()
        print("Embeddings integrated into omics data.")

        # Enhanced omics data can be saved to a file
        # output_file = 'output/enhanced_omics_data.csv'
        # enhanced_omics_data.to_csv(output_file)
        # print(f"Enhanced omics data saved to {output_file}")

        return enhanced_omics_data

    except Exception as e:
        print(f"An error occurred during the WGCNA workflow: {e}")
        raise e

def main():
    try:
        print("Starting WGCNA and GNNs Workflow...")

        omics_data = pd.DataFrame({
            'gene_feature1': [0.1, 0.2],
            'gene_feature2': [0.3, 0.4],
            'gene_feature3': [0.5, 0.6],
            'gene_feature4': [0.7, 0.8]
        }, index=['Sample1', 'Sample2'])

        phenotype_data = pd.Series([1, 0], index=['Sample1', 'Sample2'])

        clinical_data = pd.DataFrame({
            'clinical_feature1': [5, 3],
            'clinical_feature2': [7, 2]
        }, index=['Sample1', 'Sample2'])

        enhanced_omics = run_wgcna_workflow_with_gnn(omics_data, phenotype_data, clinical_data)
        print("Enhanced Omics Data:")
        print(enhanced_omics)

        print("WGCNA Workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e

if __name__ == "__main__":
    main()
