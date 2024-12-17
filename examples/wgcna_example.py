from bioneuralnet.graph_generation import WGCNA

def main():
    wgcna = WGCNA(
        phenotype_file='input/phenotype_data.csv',
        omics_list=[
            'input/genes.csv',
            'input/miRNA.csv'
        ],
        # Default values for WGCNA parameters
        data_types=['gene', 'miRNA'],        
        soft_power=6,                        
        min_module_size=30,                  
        merge_cut_height=0.25,                
        output_dir='wgcna_output_1'           
    )

    # Run WGCNA to generate adjacency matrix
    adjacency_matrix = wgcna.run()

    # Save the adjacency matrix to a CSV file
    adjacency_matrix.to_csv('wgcna_output_1/global_network.csv')

    print("Adjacency Matrix saved to 'wgcna_output_1/global_network.csv'.")

if __name__ == "__main__":
    main()
