"""
Example 1: SmCCNet Workflow
=============================

This script demonstrates how to run the SmCCNet component independently using in-memory data structures.
It constructs a network from multi-omics data and phenotype information without relying on file I/O operations.
"""

import pandas as pd
from bioneuralnet.graph_generation.smccnet import SmCCNet

def run_smccnet_workflow(omics_data: pd.DataFrame,
                         phenotype_data: pd.Series,
                         data_types: list = ['protein', 'metabolite'],
                         kfold: int = 5,
                         summarization: str = 'PCA',
                         seed: int = 732) -> pd.DataFrame:
    """
    Executes the SmCCNet-based workflow for generating an adjacency matrix.

    Args:
        omics_data (pd.DataFrame): DataFrame containing omics features (e.g., proteins, metabolites).
        phenotype_data (pd.Series): Series containing phenotype information.
        data_types (list): List of data types corresponding to omics features.
        kfold (int): Number of folds for cross-validation.
        summarization (str): Method for data summarization.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated adjacency matrix representing feature relationships.
    """
    try:
        # Step 1: Instantiate SmCCNet with in-memory data
        smccnet_instance = SmCCNet(
            phenotype_data=phenotype_data,
            omics_data=omics_data,
            data_types=data_types,
            kfold=kfold,
            summarization=summarization,
            seed=seed,
        )

        # Step 2: Run SmCCNet to generate the adjacency matrix
        adjacency_matrix = smccnet_instance.run()
        print("Adjacency matrix generated using SmCCNet.")

        return adjacency_matrix

    except Exception as e:
        print(f"An error occurred during the SmCCNet workflow: {e}")
        raise e

def main():
    try:
        print("Starting SmCCNet Workflow...")

        # Example Omics Data (Proteins and Metabolites)
        omics_data = pd.DataFrame({
            'protein_feature1': [0.8, 0.6, 0.9],
            'protein_feature2': [0.5, 0.7, 0.4],
            'metabolite_feature1': [1.2, 1.5, 1.3],
            'metabolite_feature2': [0.9, 1.1, 1.0]
        }, index=['GeneA', 'GeneB', 'GeneC'])

        # Example Phenotype Data
        phenotype_data = pd.Series([1, 0, 1], index=['GeneA', 'GeneB', 'GeneC'], name='Phenotype')

        # Run SmCCNet Workflow
        adjacency_matrix = run_smccnet_workflow(
            omics_data=omics_data,
            phenotype_data=phenotype_data
        )

        # Display the Adjacency Matrix
        print("\nGenerated Adjacency Matrix:")
        print(adjacency_matrix)

        print("\nSmCCNet Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e

if __name__ == "__main__":
    main()
