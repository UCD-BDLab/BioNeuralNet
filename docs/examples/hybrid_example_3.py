"""
Hybrid Example 3: DPMON-based Hybrid Example (SmCCNet + DPMON)

This script demonstrates a workflow where we first generate an adjacency matrix using SmCCNet,
and then use that matrix to run DPMON for disease prediction.
"""

import os
import pandas as pd
from bioneuralnet.graph_generation.smccnet import SmCCNet
from bioneuralnet.integrated_tasks import DPMON

def run_smccnet_dpmon_workflow():
    # Step 1: Generate network using SmCCNet
    smccnet_instance = SmCCNet(
        phenotype_file='input/phenotype_data.csv',
        omics_list=['input/proteins.csv', 'input/metabolites.csv'],
        data_types=['protein', 'metabolite'],
        kfold=5,
        summarization='PCA',
        seed=732,
    )
    adjacency_matrix = smccnet_instance.run()

    adjacency_output_path = os.path.join(smccnet_instance.output_dir, 'adjacency_matrix.csv')
    adjacency_matrix.to_csv(adjacency_output_path)
    print(f"Adjacency matrix saved to {adjacency_output_path}")

    # Step 2: Load data
    protein_data = pd.read_csv('input/proteins.csv', index_col=0)
    metabolite_data = pd.read_csv('input/metabolites.csv', index_col=0)
    phenotype_data = pd.read_csv('input/phenotype_data.csv', index_col=0)
    clinical_data = pd.read_csv('input/clinical_data.csv', index_col=0)

    # Step 3: Initialize and run DPMON for disease prediction
    dpmon_instance = DPMON(
        adjacency_matrix=adjacency_matrix,
        omics_list=[protein_data, metabolite_data],
        phenotype_file=phenotype_data,
        features_file=clinical_data,
        model='GCN',  # Options: 'GAT', 'SAGE', 'GIN'
        tune=False,   # Set to True for hyperparameter tuning
        gpu=False     # Set to True if GPU is available and desired
    )

    # Run DPMON
    predictions_df = dpmon_instance.run()
    if not predictions_df.empty:
        # Save predictions
        predictions_df.to_csv('dpmon_predictions.csv', index=True)
        print("DPMON workflow completed successfully. Predictions saved to dpmon_predictions.csv")
    else:
        print("DPMON hyperparameter tuning completed. No predictions were generated.")

if __name__ == "__main__":
    try:
        print("Starting SmCCNet + DPMON Hybrid Workflow...")
        run_smccnet_dpmon_workflow()
        print("Hybrid Workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e
