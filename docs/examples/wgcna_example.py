"""
Example 2: WGCNA Workflow
===========================
This script demonstrates how to run the WGCNA component independently using in-memory data structures.
It constructs a network from multi-omics data and phenotype information without relying on file I/O operations.
"""

import pandas as pd
from bioneuralnet.graph_generation.wgcna import WGCNA

def run_wgcna_workflow(omics_data: pd.DataFrame,
                       phenotype_data: pd.Series,
                       data_types: list = ['gene', 'miRNA'],
                       soft_power: int = 6,
                       min_module_size: int = 30,
                       merge_cut_height: float = 0.25) -> pd.DataFrame:
    """
    Executes the WGCNA-based workflow for generating an adjacency matrix.

    Args:
        omics_data (pd.DataFrame): DataFrame containing omics features (e.g., genes, miRNA).
        phenotype_data (pd.Series): Series containing phenotype information.
        data_types (list): List of data types corresponding to omics features.
        soft_power (int): Soft-thresholding power for WGCNA.
        min_module_size (int): Minimum module size for WGCNA.
        merge_cut_height (float): Merge cut height for WGCNA.

    Returns:
        pd.DataFrame: Generated adjacency matrix representing feature relationships.
    """
    try:
        # Step 1: Instantiate WGCNA with in-memory data
        wgcna_instance = WGCNA(
            phenotype_data=phenotype_data,
            omics_data=omics_data,
            data_types=data_types,
            soft_power=soft_power,
            min_module_size=min_module_size,
            merge_cut_height=merge_cut_height,
        )

        # Step 2: Run WGCNA to generate the adjacency matrix
        adjacency_matrix = wgcna_instance.run()
        print("Adjacency matrix generated using WGCNA.")

        return adjacency_matrix

    except Exception as e:
        print(f"An error occurred during the WGCNA workflow: {e}")
        raise e

def main():
    try:
        print("Starting WGCNA Workflow...")

        # Example Omics Data (Genes and miRNA)
        omics_data = pd.DataFrame({
            'gene_feature1': [0.1, 0.2, 0.3],
            'gene_feature2': [0.4, 0.5, 0.6],
            'miRNA_feature1': [0.7, 0.8, 0.9],
            'miRNA_feature2': [1.0, 1.1, 1.2]
        }, index=['GeneA', 'GeneB', 'GeneC'])

        # Example Phenotype Data
        phenotype_data = pd.Series([0, 1, 0], index=['GeneA', 'GeneB', 'GeneC'], name='Phenotype')

        # Run WGCNA Workflow
        adjacency_matrix = run_wgcna_workflow(
            omics_data=omics_data,
            phenotype_data=phenotype_data
        )

        # Display the Adjacency Matrix
        print("\nGenerated Adjacency Matrix:")
        print(adjacency_matrix)

        # save adjacency matrix to file
        output_file = 'output/adjacency_matrix.csv'
        adjacency_matrix.to_csv(output_file)
        print(f"Adjacency matrix saved to {output_file}")

        print("\nWGCNA Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e

if __name__ == "__main__":
    main()
