"""
SmCCNet-based Hybrid Example

This script demonstrates a workflow using SmCCNet for graph generation, followed by
embedding generation (GnnEmbedding) and subject representation (GraphEmbedding).
"""

import os
import pandas as pd
from bioneuralnet.graph_generation import SmCCNet
from bioneuralnet.network_embedding import GnnEmbedding
from bioneuralnet.subject_representation import GraphEmbedding

def run_smccnet_workflow():
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

    # Step 4: Create GraphEmbedding instance to compute node features
    subject_rep = GraphEmbedding()
    node_phenotype_corr = subject_rep.compute_node_phenotype_correlation(omics_data, phenotype_data)
    node_clinical_corr = subject_rep.compute_node_clinical_correlation(omics_data, clinical_data)
    node_features = pd.concat([node_clinical_corr, node_phenotype_corr.rename('phenotype_corr')], axis=1)

    # Step 5: Generate embeddings using GnnEmbedding
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

    # Step 6: Reduce embeddings using PCA
    node_embedding_values = subject_rep.reduce_embeddings(embeddings_df)

    # Step 7: Integrate embeddings into omics data
    enhanced_omics_data = subject_rep.run(
        adjacency_matrix=adjacency_matrix,
        omics_data=omics_data,
        phenotype_data=phenotype_data,
        clinical_data=clinical_data,
        embeddings=node_embedding_values
    )

    # Step 8: Save the enhanced omics data
    enhanced_omics_output_path = os.path.join(subject_rep.output_dir, 'enhanced_omics_data.csv')
    enhanced_omics_data.to_csv(enhanced_omics_output_path)
    print(f"Enhanced omics data saved to {enhanced_omics_output_path}")

if __name__ == "__main__":
    try:
        print("Starting SmCCNet and GNNs Workflow...")
        run_smccnet_workflow()
        print("SmCCNet Workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e
