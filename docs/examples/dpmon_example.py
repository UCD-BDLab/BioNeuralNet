import pandas as pd
from bioneuralnet.integrated_tasks import DPMON

def main():
    adjacency_matrix = pd.read_csv('input/adjacency_matrix.csv', index_col=0)
    protein_data = pd.read_csv('input/proteins.csv', index_col=0)
    metabolite_data = pd.read_csv('input/metabolites.csv', index_col=0)
    phenotype_df = pd.read_csv('input/phenotype_data.csv', index_col=0)
    clinical_data_df = pd.read_csv('input/clinical_data.csv', index_col=0)

    # Initialize DPMON
    dpmon = DPMON(
        adjacency_matrix=adjacency_matrix,
        omics_list=[protein_data, metabolite_data],
        phenotype_file=phenotype_df,
        features_file=clinical_data_df,
        model='GCN',
    )

    # Run DPMON
    predictions_df = dpmon.run()

    # If tune=False, predictions_df contains the final predictions. Print them
    if not predictions_df.empty:
        print("Disease prediction completed successfully. Sample predictions:")
        print(predictions_df.head())
    else:
        print("Hyperparameter tuning completed (no predictions generated).")

if __name__ == "__main__":
    main()

    
