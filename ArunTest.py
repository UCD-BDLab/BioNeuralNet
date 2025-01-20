"""
Example 3: Disease Prediction Using Graph Information (SmCCNet + Disease Prediction using Multi-Omics Networks (DPMON))
======================================================================================================================

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
    omics_genes: pd.DataFrame, phenotype: pd.Series, clinical_data: pd.DataFrame
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
            phenotype_df=phenotype,
            omics_dfs=[omics_genes],
            data_types=["genes"],
            kfold=5,
            summarization="PCA",
            seed=732,
        )
        adjacency_matrix = smccnet_instance.run()
        print("Adjacency matrix generated using SmCCNet.")

        dpmon_instance = DPMON(
            adjacency_matrix=adjacency_matrix,
            omics_list=[omics_genes],
            phenotype_data=phenotype,
            features_data=clinical_data,
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

        omics_genes = pd.read_csv("example_data/COPDGeneCounts.csv")
        omics_genes = omics_genes.drop(
            [
                "geneID",
                "end",
                "strand",
                "gene_id",
                "gene_name",
                "gene_type",
                "chr",
                "start",
            ],
            axis=1,
        )
        omics_genes_t = omics_genes.T
        omics_genes_t = omics_genes_t.reset_index().rename(columns={"index": "sid"})

        new_header = omics_genes_t.iloc[0].copy()
        new_header.iloc[0] = "sid"
        omics_genes_t = omics_genes_t[1:]
        omics_genes_t.columns = new_header
        omics_genes_t = omics_genes_t.reset_index(drop=True)

        # gene_names = omics_genes["geneID_split"]
        md = pd.read_csv("example_data/COPDGeneMetadata.csv")
        phenotype = md[["sid", "finalgold_visit"]].reset_index()
        clinical_data = md[
            ["sid", "age_visit", "gender", "smoking_status"]
        ].reset_index(drop=True)

        phenotype_subset = phenotype[["sid", "finalgold_visit"]]
        phenotype_subset["finalgold_visit"] = pd.to_numeric(
            phenotype_subset["finalgold_visit"], errors="coerce"
        )

        print(f"Gene name:\n {omics_genes_t}")
        print(f"Phenotype subset: \n{phenotype_subset}")
        print(f"Cclinical data: \n{clinical_data}")

        predictions = run_smccnet_dpmon_workflow(
            omics_genes_t, phenotype_subset, clinical_data
        )

        print("DPMON Predictions:")
        print(predictions)

        print("Hybrid Workflow completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the execution: {e}")
        raise e
