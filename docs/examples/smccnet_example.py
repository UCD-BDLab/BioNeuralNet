"""
Example: SmCCNet Usage
======================
This script demonstrates how to use the refactored SmCCNet class with in-memory data structures.
It generates SmCCNet-based adjacency matrices for provided phenotype and omics data represented as pandas DataFrames.
"""

import pandas as pd
from bioneuralnet.graph_generation import SmCCNet

def main():
    try:
        print("Starting SmCCNet Workflow...")

        # Example Phenotype DataFrame
        phenotype_df = pd.DataFrame({
            'SampleID': ['S1', 'S2', 'S3', 'S4'],
            'Phenotype': ['Control', 'Treatment', 'Control', 'Treatment']
        })

        # Example Omics DataFrames
        omics_df1 = pd.DataFrame({
            'SampleID': ['S1', 'S2', 'S3', 'S4'],
            'GeneA': [1.2, 2.3, 3.1, 4.0],
            'GeneB': [2.1, 3.4, 1.2, 3.3],
            'GeneC': [3.3, 1.5, 2.2, 4.1]
        })

        omics_df2 = pd.DataFrame({
            'SampleID': ['S1', 'S2', 'S3', 'S4'],
            'GeneD': [4.2, 5.3, 6.1, 7.0],
            'GeneE': [5.1, 6.4, 4.2, 6.3],
            'GeneF': [6.3, 4.5, 5.2, 7.1]
        })

        # List of omics DataFrames and their data types
        omics_dfs = [omics_df1, omics_df2]
        data_types = ['Transcriptomics', 'Proteomics']

        # Initialize SmCCNet Instance
        smccnet = SmCCNet(
            phenotype_df=phenotype_df,
            omics_dfs=omics_dfs,
            data_types=data_types,
            kfold=5,
            summarization="PCA",
            seed=732
        )

        # Run SmCCNet
        adjacency_matrix = smccnet.run()

        # Display Adjacency Matrix
        print("\nAdjacency Matrix:")
        print(adjacency_matrix)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e

if __name__ == "__main__":
    main()
