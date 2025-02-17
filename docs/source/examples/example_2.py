"""
Example 2: Disease Prediction Using Graph Information (SmCCNet + DPMON)
=======================================================================

This script demonstrates a workflow where we first generate a graph using Sparse Multiple Canonical Correlation Network
(SmCCNet), and then use that network matrix to run Disease Prediction using Multi-Omics Networks (DPMON), a pipeline
that leverages Graph Neural Networks (GNNs) for disease phenotype prediction.

Steps:
1. Load synthetic multi-omics and phenotype data using DatasetLoader.
2. Generate an adjacency matrix using SmCCNet based on multi-omics and phenotype data.
3. Utilize DPMON to predict disease phenotypes using the network information and omics data.
"""

import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.downstream_task import DPMON

def run_smccnet_dpmon_workflow() -> pd.DataFrame:
    """
    Executes the hybrid workflow combining SmCCNet for network generation and DPMON for disease prediction.

    Steps:
        1. Loads synthetic dataset (omics, phenotype, and clinical data).
        2. Generates an adjacency matrix using SmCCNet.
        3. Runs DPMON for disease prediction based on the adjacency matrix.

    Returns:
        pd.DataFrame: Disease prediction results from DPMON.
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

        # Run Disease Prediction using DPMON
        print("Running DPMON...")
        dpmon = DPMON(
            adjacency_matrix=global_network,
            omics_list=[omics1, omics2],
            phenotype_data=phenotype,
            clinical_data=clinical,
            tune=True,
        )

        dpmon_predictions = dpmon.run()

        if not dpmon_predictions.empty:
            print("DPMON workflow completed successfully. Predictions generated.")
        else:
            print("DPMON hyperparameter tuning completed. No predictions were generated.")

        return dpmon_predictions

    except Exception as e:
        print(f"An error occurred during the SmCCNet + DPMON workflow: {e}")
        raise e


if __name__ == "__main__":
    try:
        print("Starting SmCCNet + DPMON Workflow...")

        predictions = run_smccnet_dpmon_workflow()

        print("\nDPMON Predictions:")
        print(predictions.head())

        print("\nHybrid Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e
