import os
import pandas as pd

class DatasetLoader:
    def __init__(self, dataset_name: str):
        """
        Initializes the loader with the dataset name.
        
        Args:
            dataset_name (str): Either "example1" or "example2".
        """
        self.dataset_name = dataset_name.strip().lower()
        self.base_dir = os.path.dirname(__file__)
    
    def load_data(self):
        """
        Loads the dataset and returns a tuple of four DataFrames.
        
        Returns:
            tuple: 
              - For "example1": (omics1, omics2, pheno, clinical)
              - For "example2": (gene_data, mirna_data, rppa_data, clinical_data)
        
        Raises:
            FileNotFoundError: If any required file is missing.
            ValueError: If the dataset name is not valid.
        """
        if self.dataset_name == "example2":
            dataset_path = os.path.join(self.base_dir, "example2")
            gene_file     = os.path.join(dataset_path, "gene_data.csv")
            mirna_file    = os.path.join(dataset_path, "mirna_data.csv")
            rppa_file     = os.path.join(dataset_path, "rppa_data.csv")
            clinical_file = os.path.join(dataset_path, "clinical_data.csv")
            
            for f in [gene_file, mirna_file, rppa_file, clinical_file]:
                if not os.path.isfile(f):
                    raise FileNotFoundError(
                        f"Required file '{os.path.basename(f)}' not found in '{dataset_path}'."
                    )
            
            gene_data    = pd.read_csv(gene_file)
            mirna_data   = pd.read_csv(mirna_file)
            rppa_data    = pd.read_csv(rppa_file)
            clinical_data = pd.read_csv(clinical_file)
            
            return gene_data, mirna_data, rppa_data, clinical_data
        
        elif self.dataset_name == "example1":
            dataset_path = os.path.join(self.base_dir, "example1")
            x1_file      = os.path.join(dataset_path, "X1.csv")
            x2_file      = os.path.join(dataset_path, "X2.csv")
            y_file       = os.path.join(dataset_path, "Y.csv")
            clinical_file = os.path.join(dataset_path, "clinical_data.csv")

            for f in [x1_file, x2_file, y_file, clinical_file]:
                if not os.path.isfile(f):
                    raise FileNotFoundError(
                        f"Required file '{os.path.basename(f)}' not found in '{dataset_path}'."
                    )
            
            omics1 = pd.read_csv(x1_file, index_col=0)
            omics2 = pd.read_csv(x2_file, index_col=0)
            pheno  = pd.read_csv(y_file, index_col=0)
            clinical = pd.read_csv(clinical_file, index_col=0)  
            return omics1, omics2, pheno, clinical
        
        else:
            raise ValueError("Dataset name must be either 'example1' or 'example2'.")
