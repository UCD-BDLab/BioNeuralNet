import os
import pandas as pd

class DatasetLoader:
    def __init__(self, dataset_name: str):
        """
        Initializes the loader with the dataset name.
        
        Attributes:

            dataset_name (str): "example1"
        """
        self.dataset_name = dataset_name.strip().lower()
        self.base_dir = os.path.dirname(__file__)
    
    def load_data(self):
        """
        Loads the dataset and returns a tuple.
        
        Returns:

            tuple: 
            
                For "example1": (omics1, omics2, pheno, clinical)
        
        Raises:

            FileNotFoundError: If any required file is missing.
            ValueError: If the dataset name is not valid.
            
        """    
        if self.dataset_name == "example1":
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
    