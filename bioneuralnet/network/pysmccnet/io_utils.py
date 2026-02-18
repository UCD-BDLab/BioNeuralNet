import pandas as pd
import os

def load_r_export_folds(base_path, num_omics, k=5):
    """
    Loads the specific SmCCNet directory structure exported from R.
    Structure: base_path/fold_N/X_train_Omics_M.csv
    """
    folddata = {}
    print(f"Loading R-exported folds from: {base_path}")
    
    for i in range(1, k + 1):
        fold_key = f"fold_{i}"
        fold_dir = os.path.join(base_path, fold_key)
        
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Could not find directory: {fold_dir}")
            
        x_train_list = []
        x_test_list = []
        
        for omics_idx in range(1, num_omics + 1):
            xtrain_path = os.path.join(fold_dir, f"X_train_Omics_{omics_idx}.csv")
            xtest_path = os.path.join(fold_dir, f"X_test_Omics_{omics_idx}.csv")
            
            x_train = pd.read_csv(xtrain_path).to_numpy()
            x_test = pd.read_csv(xtest_path).to_numpy()
            
            x_train_list.append(x_train)
            x_test_list.append(x_test)
            
        ytrain_path = os.path.join(fold_dir, "Y_train.csv")
        ytest_path = os.path.join(fold_dir, "Y_test.csv")
        
        y_train = pd.read_csv(ytrain_path).iloc[:, 0].to_numpy()
        y_test = pd.read_csv(ytest_path).iloc[:, 0].to_numpy()
        
        folddata[fold_key] = {
            "X_train": x_train_list,
            "X_test": x_test_list,
            "Y_train": y_train,
            "Y_test": y_test
        }
        
    print(f"Successfully loaded {k} folds.")
    return folddata