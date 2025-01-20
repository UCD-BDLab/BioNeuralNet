"""
Example 2: Disease Prediction Using Graph Information (SmCCNet + Disease Prediction using Multi-Omics Networks (DPMON))
========================================================================================================================

This script demonstrates a workflow where we first generate a graph using Sparse Multiple Canonical Correlation Network
(SmCCNet), and then use that network matrix to run Disease Prediction using Multi-Omics Networks (DPMON), a pipeline
that leverages the power of Graph Neural Networks (GNNs) specifically designed to predict disease phenotypes.

Steps:
1. Generate an adjacency matrix using SmCCNet based on multi-omics and phenotype data.
2. Utilize DPMON to predict disease phenotypes using the network information and omics data.
"""

import pandas as pd
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.downstream_task import DPMON


def run_smccnet_dpmon_workflow(
    omics_proteins: pd.DataFrame,
    omics_metabolites: pd.DataFrame,
    phenotype_data: pd.DataFrame,
    clinical_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executes the hybrid workflow combining SmCCNet for network generation and DPMON for disease prediction.

    This function performs the following steps:
        1. Generates an adjacency matrix using SmCCNet.
        2. Initializes and runs DPMON for disease prediction based on the adjacency matrix.
        3. Returns the disease prediction results.

    Args:
        omics_proteins (pd.DataFrame): DataFrame containing protein data.
        omics_metabolites (pd.DataFrame): DataFrame containing metabolite data.
        phenotype_data (pd.Series): Series containing phenotype information.
        clinical_data (pd.DataFrame): DataFrame containing clinical data.

    Returns:
        pd.DataFrame: Disease prediction results from DPMON.
    """
    try:
        smccnet_instance = SmCCNet(
            phenotype_df=phenotype_data,
            omics_dfs=[omics_proteins, omics_metabolites],
            data_types=["protein", "metabolite"],
            kfold=5,
            summarization="PCA",
            seed=732,
        )
        adjacency_matrix = smccnet_instance.run()
        print("Adjacency matrix generated using SmCCNet.")

        dpmon_instance = DPMON(
            adjacency_matrix=adjacency_matrix,
            omics_list=[omics_proteins, omics_metabolites],
            phenotype_data=phenotype_data,
            clinical_data=clinical_data,
            model="GCN",
            tune=False,
            gpu=False,
        )

        predictions_df = dpmon_instance.run()
        if not predictions_df.empty:
            print("DPMON workflow completed successfully. Predictions generated.")
        else:
            print(
                "DPMON hyperparameter tuning completed. No predictions were generated."
            )

        return predictions_df

    except Exception as e:
        print(f"An error occurred during the SmCCNet + DPMON workflow: {e}")
        raise e


if __name__ == "__main__":
    try:
        print("Starting SmCCNet + DPMON Hybrid Workflow...")

        omics_proteins = pd.DataFrame(
            {"protein_feature1": [0.1, 0.2], "protein_feature2": [0.3, 0.4]},
            index=["Sample1", "Sample2"],
        )

        omics_metabolites = pd.DataFrame(
            {"metabolite_feature1": [0.5, 0.6], "metabolite_feature2": [0.7, 0.8]},
            index=["Sample1", "Sample2"],
        )

        phenotype_data = pd.Series([1, 0], index=["Sample1", "Sample2"])

        clinical_data = pd.DataFrame(
            {"clinical_feature1": [5, 3], "clinical_feature2": [7, 2]},
            index=["Sample1", "Sample2"],
        )

        predictions = run_smccnet_dpmon_workflow(
            omics_proteins, omics_metabolites, phenotype_data, clinical_data
        )

        print("DPMON Predictions:")
        print(predictions)

        print("Hybrid Workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e
