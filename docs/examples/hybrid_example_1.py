"""
Hybrid Example 1: SmCCNet Workflow with GNN Embeddings
=======================================================

This tutorial demonstrates how to perform a comprehensive workflow using SmCCNet for graph generation, followed by
GNN-based embedding generation and subject representation integration.
"""

import os
import pandas as pd
from bioneuralnet.graph_generation.smccnet import SmCCNet
from bioneuralnet.network_embedding import GnnEmbedding
from bioneuralnet.subject_representation import GraphEmbedding

def run_smccnet_workflow():
    """
    Executes the SmCCNet-based workflow for generating enhanced omics data.

    This function performs the following steps:
        1. Instantiates the SmCCNet, GnnEmbedding, and GraphEmbedding components.
        2. Loads omics, phenotype, and clinical data.
        3. Generates an adjacency matrix using SmCCNet.
        4. Computes node features based on correlations.
        5. Generates embeddings using GnnEmbedding.
        6. Reduces embeddings using PCA.
        7. Integrates embeddings into omics data to produce enhanced omics data.
        8. Saves the enhanced omics data to the output directory.
    """
    try:
        # Step 1: Instantiate SmCCNet
        smccnet_instance = SmCCNet(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/proteins.csv', 'input/metabolites.csv'],
            data_types=['protein', 'metabolite'],
            kfold=5,
            summarization='PCA',
            seed=732,
        )

        # Step 2: Load data as DataFrames
        omics_data = pd.read_csv('input/omics_data.csv', index_col=0)
        phenotype_data = pd.read_csv('input/phenotype_data.csv', index_col=0).squeeze()
        clinical_data = pd.read_csv('input/clinical_data.csv', index_col=0)

        # Step 3: Generate adjacency matrix using SmCCNet
        adjacency_matrix = smccnet_instance.run()
        adjacency_output_path = os.path.join(smccnet_instance.output_dir, 'adjacency_matrix.csv')
        adjacency_matrix.to_csv(adjacency_output_path)
        print(f"Adjacency matrix saved to {adjacency_output_path}")

        # Step 4: Initialize and run GnnEmbedding
        node_features = pd.concat([
            omics_data[['protein_feature1', 'protein_feature2']],  # Replace with actual feature names
            omics_data[['metabolite_feature1', 'metabolite_feature2']]  # Replace with actual feature names
        ], axis=1)

        gnn_embedding = GnnEmbedding(
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

        # Step 5: Initialize and run GraphEmbedding
        graph_embedding = GraphEmbedding(
            adjacency_matrix=adjacency_matrix,
            omics_data=omics_data,
            phenotype_data=phenotype_data,
            clinical_data=clinical_data,
            embedding_method='GNNs'
        )
        enhanced_omics_data = graph_embedding.run()

        # Step 6: Save the enhanced omics data
        enhanced_omics_output_path = os.path.join(graph_embedding.output_dir, 'enhanced_omics_data.csv')
        enhanced_omics_data.to_csv(enhanced_omics_output_path)
        print(f"Enhanced omics data saved to {enhanced_omics_output_path}")

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
        raise fnf_error
    except Exception as e:
        print(f"An error occurred during the SmCCNet workflow: {e}")
        raise e

if __name__ == "__main__":
    try:
        print("Starting SmCCNet and GNNs Workflow...")
        run_smccnet_workflow()
        print("SmCCNet Workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e
