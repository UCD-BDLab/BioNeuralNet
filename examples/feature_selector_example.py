import pandas as pd
from bioneuralnet.analysis.feature_selector import FeatureSelector
from bioneuralnet.subject_representation.subject_representation import GraphEmbedding

def main():
    omics_files = ['input/genetic_data.csv', 'input/protein_data.csv', 'input/metabolite_data.csv'] 
    phenotype_file = 'input/phenotype_data.csv'  
    clinical_data_file = 'input/clinical_data.csv'  
    adjacency_matrix_file = 'input/adjacency_matrix.csv'  

    adjacency_matrix = pd.read_csv(adjacency_matrix_file, index_col=0)

    subject_rep = GraphEmbedding(
        adjacency_matrix=adjacency_matrix,
        omics_list=omics_files,
        phenotype_file=phenotype_file,
        clinical_data_file=clinical_data_file,
        embedding_method='GNNs',  
    )
    enhanced_omics_data = subject_rep.run()

    phenotype_data = pd.read_csv(phenotype_file, index_col=0).iloc[:, 0]

    feature_selector = FeatureSelector(
        enhanced_omics_data=enhanced_omics_data,
        phenotype_data=phenotype_data,
        num_features=20,  
        selection_method='lasso', 
    )
    selected_genetic_features = feature_selector.run_feature_selection()

    print("Selected Multi-Omics Features:")
    print(selected_genetic_features.head())

if __name__ == "__main__":
    main()
