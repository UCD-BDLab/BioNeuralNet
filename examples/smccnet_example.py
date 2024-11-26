from bioneuralnet.graph_generation.smccnet import SmCCNet

def main():
    smccnet = SmCCNet(
        phenotype_file='input/phenotype_data.csv',
        omics_list=[
            'input/proteins.csv',
            'input/metabolites.csv'
        ],
        data_types=['protein', 'metabolite'],  
        kfold=5,                              
        summarization='PCA',                  
        seed=732,                              
    )

    adjacency_matrix = smccnet.run()
    adjacency_matrix.to_csv('smccnet_output_1/global_network.csv')

    print("Adjacency Matrix saved to 'smccnet_output_1/global_network.csv'.")

if __name__ == "__main__":
    main()
