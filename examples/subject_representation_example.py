import pandas as pd
from bioneuralnet.subject_representation.subject_representation import SubjectRepresentationEmbedding

def main():
    # Paths to input files
    omics_files = ['input/proteins.csv', 'input/metabolites.csv']
    phenotype_file = 'input/phenotype_data.csv'
    clinical_data_file = 'input/clinical_data.csv'
    adjacency_matrix_file = 'input/adjacency_matrix.csv'

    # Load adjacency matrix
    adjacency_matrix = pd.read_csv(adjacency_matrix_file, index_col=0)

    # Initialize SubjectRepresentationEmbedding
    subject_rep_embedding = SubjectRepresentationEmbedding(
        adjacency_matrix=adjacency_matrix,
        omics_list=omics_files,
        phenotype_file=phenotype_file,
        clinical_data_file=clinical_data_file,
        embedding_method='GNNs'
    )

    # Run the subject representation process
    enhanced_omics_data = subject_rep_embedding.run()

    # The enhanced omics data is saved to the output directory specified in the class
    print("Subject representation workflow completed successfully.")


if __name__ == "__main__":
    main()
