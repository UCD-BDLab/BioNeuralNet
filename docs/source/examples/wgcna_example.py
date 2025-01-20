import pandas as pd
from bioneuralnet.external_tools import WGCNA


def run_wgcna_workflow(
    omics_data: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    data_types: list = ["gene", "miRNA"],
    soft_power: int = 6,
    min_module_size: int = 30,
    merge_cut_height: float = 0.25,
) -> pd.DataFrame:
    try:
        wgcna_instance = WGCNA(
            phenotype_df=phenotype_df,
            omics_dfs=omics_data,
            data_types=data_types,
            soft_power=soft_power,
            min_module_size=min_module_size,
            merge_cut_height=merge_cut_height,
        )

        adjacency_matrix = wgcna_instance.run()
        print("Adjacency matrix generated using WGCNA.")

        return adjacency_matrix

    except Exception as e:
        print(f"An error occurred during the WGCNA workflow: {e}")
        raise e


def main():
    try:
        print("Starting WGCNA Workflow...")

        omics_data = pd.DataFrame(
            {
                "gene_feature1": [0.1, 0.2, 0.3],
                "gene_feature2": [0.4, 0.5, 0.6],
                "miRNA_feature1": [0.7, 0.8, 0.9],
                "miRNA_feature2": [1.0, 1.1, 1.2],
            },
            index=["GeneA", "GeneB", "GeneC"],
        )

        phenotype_data = pd.DataFrame(
            [0, 1, 0], index=["GeneA", "GeneB", "GeneC"], name="Phenotype"
        )
        adjacency_matrix = run_wgcna_workflow(
            omics_data=omics_data, phenotype_data=phenotype_data
        )

        print("\nGenerated Adjacency Matrix:")
        print(adjacency_matrix)

        output_file = "output/adjacency_matrix.csv"
        adjacency_matrix.to_csv(output_file)

        print(f"Adjacency matrix saved to {output_file}")
        print("\nWGCNA Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e


if __name__ == "__main__":
    main()
