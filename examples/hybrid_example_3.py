"""
DPMON-based Hybrid Example

This script demonstrates a workflow using DPMON for disease prediction after network embedding.
It shows how to use DPMON end-to-end, given that DPMON internally uses GNN embeddings,
node feature correlations, and downstream prediction.
"""

import pandas as pd
from bioneuralnet.integrated_tasks import DPMON

def run_dpmon_workflow():
    # Load data as DataFrames
    adjacency_matrix = pd.read_csv('input/adjacency_matrix.csv', index_col=0)
    protein_data = pd.read_csv('input/proteins.csv', index_col=0)
    metabolite_data = pd.read_csv('input/metabolites.csv', index_col=0)
    phenotype_data = pd.read_csv('input/phenotype_data.csv', index_col=0)
    clinical_data = pd.read_csv('input/clinical_data.csv', index_col=0)

    # Initialize DPMON
    dpmon_instance = DPMON(
        adjacency_matrix=adjacency_matrix,
        omics_list=[protein_data, metabolite_data],
        phenotype_file=phenotype_data,
        features_file=clinical_data,
        model='GCN',  # or GAT, SAGE, GIN
        tune=False,   # Set to True if you want hyperparameter tuning
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
        print("Starting DPMON Workflow...")
        run_dpmon_workflow()
        print("DPMON Workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e
