"""
Main SmCCNet pipeline. Supports both CCA (continuous) and PLS (binary) phenotypes.
"""

import os
import torch
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Union

from .math_helpers import r_scale_torch, r_scale
from .analysis import data_preprocess, get_can_cor_multi, get_abar, get_omics_modules, prune_modules
from .wrappers import get_can_weights_multi,get_robust_weights_multi,get_robust_weights_multi_binary

from ...utils.logger import get_logger
logger = get_logger(__name__)

def auto_pysmccnet(X: List[Union[pd.DataFrame, np.ndarray]], Y: Union[pd.DataFrame, np.ndarray], AdjustedCovar: Optional[pd.DataFrame] = None, preprocess: bool = False, Kfold: int = 5, subSampNum: int = 100, DataType: Optional[List[str]] = None, BetweenShrinkage: float = 2.0, ScalingPen: List[float] = [0.1, 0.1], saving_dir: str = os.getcwd(), tuneLength: int = 5, tuneRangeCCA: List[float] = [0.1, 0.5], tuneRangePLS: List[float] = [0.5, 0.9], EvalMethod: str = 'accuracy', ncomp_pls: int = 3, seed: int = 123, CutHeight: float = 1 - 0.1**10, min_size: int = 10, max_size: int = 100, summarization: str = "NetSHy", precomputed_fold_data: Optional[dict] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64) -> dict:
    """Automated SmCCNet workflow with GPU acceleration.

    Runs the complete SmCCNet pipeline supporting both CCA (continuous phenotype) and PLS (binary phenotype) modes. The workflow includes optional preprocessing, cross-validation for penalty tuning, subsampling for stability selection, and final network construction.

    Args:

        X (List[pd.DataFrame | np.ndarray]): Input data matrices (omics layers) for integration.
        Y (pd.DataFrame | np.ndarray): Phenotype vector; numeric for CCA or binary (0/1) for PLS.
        AdjustedCovar (pd.DataFrame | None): Optional covariates to regress out from X before analysis.
        preprocess (bool): If True, center and scale data; if False, use raw input.
        Kfold (int): Number of cross-validation folds for penalty parameter tuning.
        subSampNum (int): Number of subsampling iterations for stability selection.
        DataType (List[str] | None): Names for each omics layer in X; defaults to generic names if None.
        BetweenShrinkage (float): Shrinkage factor for between-omics scaling weights.
        ScalingPen (List[float]): Penalty terms used for determining scaling factors.
        saving_dir (str): Directory path for saving output results.
        tuneLength (int): Number of candidate penalty parameters to test per omics layer.
        tuneRangeCCA (List[float]): Min and max penalty values for CCA (continuous phenotype).
        tuneRangePLS (List[float]): Min and max penalty values for PLS (binary phenotype).
        EvalMethod (str): Metric for PLS evaluation; one of 'accuracy', 'auc', 'precision', 'recall', or 'f1'.
        ncomp_pls (int): Number of latent components to use for PLS models.
        CutHeight (float): Height threshold for hierarchical tree cutting in module extraction.
        min_size (int): Minimum number of nodes to retain a network module.
        max_size (int): Maximum module size; larger modules are pruned down.
        summarization (str): Network summarization method. Currently only 'NetSHy' is supported.
        seed (int): Random seed for reproducibility.
        precomputed_fold_data (dict | None): Precomputed CV folds to bypass internal fold generation.
        device (torch.device | None): PyTorch device; if None, automatically selects GPU if available.
        dtype (torch.dtype): PyTorch data type for computations.

    Returns:

        dict: Dictionary containing results for 'CCA' or 'PLS' including adjacency matrices, processed data, and CV results.

    """
    np.random.seed(seed)
    
    # device configuration
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n")
    print("**********************************")
    print("* Welcome to Automated SmCCNet! *")
    print(f"* Device: {device}")
    print("**********************************")
    print("\n")

    if not isinstance(X, list):
        X = [X]
    
    if DataType is None:
        DataType = [f"Omics{i+1}" for i in range(len(X))]

    feature_labels = []
    
    for i, data_obj in enumerate(X):
        prefix = DataType[i]
        
        if hasattr(data_obj, 'columns'):
            # DataFrame: use real column names with DataType prefix
            labels = [f"{prefix}_{col}" for col in data_obj.columns]
        else:
            # Numpy array: fallback to generic
            labels = [f"{prefix}_Feat{j+1}" for j in range(data_obj.shape[1])]
        
        feature_labels.extend(labels)

    # preprocessing
    if preprocess:
        print("\n--------------------------------------------------")
        print(">> Starting data preprocessing...")
        print("--------------------------------------------------\n")
        
        if AdjustedCovar is not None:
            print("Covariate(s) provided. Regressing out effects.\n")
            AdjustedCovar = pd.DataFrame(AdjustedCovar)
            
        X_processed = []
        for xx in range(len(X)):
            x_df = pd.DataFrame(X[xx])
            processed = data_preprocess(X=x_df, covariates=AdjustedCovar, is_cv=False, center=True, scale=True, device=device, dtype=dtype)
            X_processed.append(processed)
        
        X = [x.to_numpy() for x in X_processed]
    else:
        X = [np.array(x) for x in X]

    Y = np.array(Y)

    # determine analysis type and method
    if len(X) <= 1:
        raise ValueError("User must provide a list of data matrices (multiple omics).")
    
    AnalysisType = "multiomics"
    
    # method selection: CCA for continuous, PLS for binary
    if Y.ndim > 1 and Y.shape[1] > 1:
        method = "CCA"
    elif _is_binary(Y):
        method = "PLS"
    else:
        method = "CCA"
    
    print(f"This project uses {AnalysisType} {method}\n")

    # multi-omics scaling factor
    ScalingFactor = None
    ScalingFactorCC = None
    BDRatio = None
    
    print("\n--------------------------------------------------")
    print(">> Determining scaling factor...")
    print("--------------------------------------------------\n")

    # between-omics pairs
    comb_indices = list(itertools.combinations(range(len(X)), 2))
    ScalingFactor_Omics = np.zeros(len(comb_indices))
    ScalingFactorNames = []
    DataTypePheno = list(DataType) + ["Phenotype"]

    for i, (idx1, idx2) in enumerate(comb_indices):
        name = f"{DataTypePheno[idx1]}-{DataTypePheno[idx2]}"
        ScalingFactorNames.append(name)
        
        X_pair = [
            torch.tensor(X[idx1], device=device, dtype=dtype),
            torch.tensor(X[idx2], device=device, dtype=dtype)
        ]
        
        CC_weight = get_can_weights_multi(X_pair, Trait=None, Lambda=ScalingPen, no_trait=True)
        
        proj1 = torch.matmul(X_pair[0], CC_weight[0])
        proj2 = torch.matmul(X_pair[1], CC_weight[1])
        
        p1 = proj1.flatten()
        p2 = proj2.flatten()
        p1_c = p1 - torch.mean(p1)
        p2_c = p2 - torch.mean(p2)
        num = torch.sum(p1_c * p2_c)
        den = torch.sqrt(torch.sum(p1_c ** 2)) * torch.sqrt(torch.sum(p2_c ** 2))
        corr_val = (num / den).item() if den.item() != 0 else 0.0
        
        ScalingFactor_Omics[i] = abs(corr_val)

    # shrink between-omics scaling factor
    ScalingFactor_Omics = ScalingFactor_Omics / BetweenShrinkage

    if method == 'PLS':
        BDRatio = [np.mean(ScalingFactor_Omics), 1.0]
        print(f"The between-omics and omics-phenotype importance ratio is: {BDRatio[0]:.4f}:{BDRatio[1]}\n")
    
    # build full scaling factor vector
    ScalingFactorCC = np.concatenate([ScalingFactor_Omics, np.ones(len(X))])
    
    K_total = len(X) + 1
    comb_total = list(itertools.combinations(range(K_total), 2))
    
    # non-phenotype index
    non_pheno_index = [i for i, pair in enumerate(comb_total) if pair[1] != len(X)]
    
    ScalingFactor_full = np.ones(len(comb_total))
    omics_pairs_lookup = {pair: val for pair, val in zip(comb_indices, ScalingFactor_Omics)}
    
    for i, pair in enumerate(comb_total):
        if pair in omics_pairs_lookup:
            ScalingFactor_full[i] = omics_pairs_lookup[pair]
    
    if method == 'PLS':
        ScalingFactor = ScalingFactor_full[non_pheno_index]
    else:
        ScalingFactor = ScalingFactor_full

    print("The scaling factor selection is: \n")
    
    # print all scaling factors
    all_names = [f"{DataTypePheno[pair[0]]}-{DataTypePheno[pair[1]]}" for pair in comb_total]
    if method == 'PLS':
        for idx in non_pheno_index:
            print(f"{all_names[idx]}: {ScalingFactor_full[idx]}")
    else:
        for i, val in enumerate(ScalingFactor_full):
            print(f"{all_names[i]}: {val}")

    # penalty selection (CV)
    print("\n--------------------------------------------------")
    print(">> Determining best penalty via Cross-Validation...")
    print("--------------------------------------------------\n")

    if method == 'CCA':
        pen_seq = np.linspace(tuneRangeCCA[0], tuneRangeCCA[1], tuneLength)
        grid_inputs = [pen_seq for _ in range(len(X))]
        PenComb = np.array(list(itertools.product(*grid_inputs)))
    elif method == 'PLS':
        pen_seq = np.linspace(tuneRangePLS[0], tuneRangePLS[1], tuneLength)
        grid_inputs = [pen_seq for _ in range(len(X) + 1)]
        PenComb = np.array(list(itertools.product(*grid_inputs)))
    
    n_grid = PenComb.shape[0]

    SubsamplingPercent = np.zeros(len(X))
    for i in range(len(X)):
        SubsamplingPercent[i] = 0.9 if X[i].shape[1] < 300 else 0.7

    # prepare cv data
    if precomputed_fold_data is not None:
        print(">> Using PRECOMPUTED folds from SmCCNet-R export...")
        if method == 'CCA':
            folddata = {}
            for fold_key, fold_val in precomputed_fold_data.items():
                folddata[fold_key] = {
                    "X_train": [torch.tensor(x, device=device, dtype=dtype) for x in fold_val["X_train"]],
                    "X_test": [torch.tensor(x, device=device, dtype=dtype) for x in fold_val["X_test"]],
                    "Y_train": torch.tensor(fold_val["Y_train"], device=device, dtype=dtype),
                    "Y_test": torch.tensor(fold_val["Y_test"], device=device, dtype=dtype)
                }
            folddata_cpu = precomputed_fold_data
        else:
            # PLS: keep data as numpy
            folddata = precomputed_fold_data
            folddata_cpu = precomputed_fold_data
    else:
        print(">> Generating random K-Fold splits...")
        
        # scale data
        if method == 'CCA':
            X_scaled = [r_scale_torch(torch.tensor(x, device=device, dtype=dtype)) for x in X]
            Y_scaled = r_scale_torch(torch.tensor(Y, device=device, dtype=dtype))
        else:
            # PLS: scale X, keep Y binary
            X_scaled_np = [r_scale(x) for x in X]
            Y_binary = Y.flatten()
        
        n_samples = X[0].shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_indices = np.array_split(indices, Kfold)
        
        folddata = {}
        folddata_cpu = {}
        
        for k in range(Kfold):
            test_idx = np.sort(fold_indices[k])
            train_idx = np.sort(np.setdiff1d(indices, test_idx))
            
            if method == 'CCA':
                folddata[f"fold_{k+1}"] = {
                    "X_train": [x[train_idx, :] for x in X_scaled],
                    "X_test": [x[test_idx, :] for x in X_scaled],
                    "Y_train": Y_scaled[train_idx],
                    "Y_test": Y_scaled[test_idx]
                }
                folddata_cpu[f"fold_{k+1}"] = {
                    "X_train": [x[train_idx, :].cpu().numpy() for x in X_scaled],
                    "X_test": [x[test_idx, :].cpu().numpy() for x in X_scaled],
                    "Y_train": Y_scaled[train_idx].cpu().numpy(),
                    "Y_test": Y_scaled[test_idx].cpu().numpy()
                }
            else:
                # PLS: numpy throughout
                folddata[f"fold_{k+1}"] = {
                    "X_train": [x[train_idx, :] for x in X_scaled_np],
                    "X_test": [x[test_idx, :] for x in X_scaled_np],
                    "Y_train": Y_binary[train_idx],
                    "Y_test": Y_binary[test_idx]
                }
                folddata_cpu = folddata

    # cv loop: CCA
    if method == 'CCA':
        CVResult_list = []
        
        for k in range(Kfold):
            fold_key = f"fold_{k+1}"
            d = folddata[fold_key]
            
            RhoTrain = np.zeros(n_grid)
            RhoTest = np.zeros(n_grid)
            DeltaCor = np.zeros(n_grid)
            
            print(f"Processing fold {k+1}/{Kfold}...")

            for idx in range(n_grid):
                l1 = PenComb[idx, :]
                
                Y_train = d["Y_train"]
                if isinstance(Y_train, torch.Tensor) and Y_train.ndim == 1:
                    Y_train = Y_train.reshape(-1, 1)
                Y_test = d["Y_test"]
                if isinstance(Y_test, torch.Tensor) and Y_test.ndim == 1:
                    Y_test = Y_test.reshape(-1, 1)
                
                Ws = get_can_weights_multi(X=d["X_train"], Trait=Y_train, Lambda=l1, no_trait=False, cc_coef=ScalingFactor)
                
                rho_train = get_can_cor_multi(X=d["X_train"], Y=Y_train, cc_weight=Ws, cc_coef=ScalingFactorCC)
                
                rho_test = get_can_cor_multi(X=d["X_test"], Y=Y_test, cc_weight=Ws, cc_coef=ScalingFactorCC)
                
                RhoTrain[idx] = round(rho_train, 5)
                RhoTest[idx] = round(rho_test, 5)
                DeltaCor[idx] = abs(rho_train - rho_test)
                
            CVResult_df = pd.DataFrame({
                "RhoTrain": RhoTrain,
                "RhoTest": RhoTest,
                "DeltaCor": DeltaCor
            })
            CVResult_list.append(CVResult_df)

        AggregatedCVResult = sum(CVResult_list) / len(CVResult_list)
        
        epsilon = 1e-10
        EvalMetric = AggregatedCVResult["DeltaCor"] / (np.abs(AggregatedCVResult["RhoTest"]) + epsilon)
        best_idx = EvalMetric.idxmin()
        BestPen = PenComb[best_idx, :]
        
        print("\n")
        for xx in range(len(BestPen)):
            print(f"The best penalty term for {DataType[xx]} is: {BestPen[xx]}")
        
        best_rho = round(AggregatedCVResult.iloc[best_idx]["RhoTest"], 3)
        best_err = round(AggregatedCVResult.iloc[best_idx]["DeltaCor"], 3)
        print(f"Testing Canonical Correlation: {best_rho}, Prediction Error: {best_err}\n")
        
        # final run: CCA
        print("Running multi-omics CCA with best penalty on complete dataset.\n")
        
        X_scaled_final = [r_scale_torch(torch.tensor(x, device=device, dtype=dtype)) for x in X]
        Y_scaled_final = r_scale_torch(torch.tensor(Y, device=device, dtype=dtype))
        
        Ws_final = get_robust_weights_multi(X=X_scaled_final, Trait=Y_scaled_final, no_trait=False, cc_coef=ScalingFactor, Lambda=BestPen, s=SubsamplingPercent, subsampling_num=subSampNum)

    # cv loop: PLS
    elif method == 'PLS':
        import statsmodels.api as sm
        
        CVResult_list = []
        
        for k in range(Kfold):
            fold_key = f"fold_{k+1}"
            d = folddata[fold_key]
            
            TrainMetric = np.zeros(n_grid)
            TestMetric = np.zeros(n_grid)
            
            print(f"Processing fold {k+1}/{Kfold}...")
            
            for idx in range(n_grid):
                l1 = PenComb[idx, :]
                
                lambda_between = l1[:len(X)]
                lambda_pheno = l1[len(X)]
                
                X_train_torch = [torch.tensor(x, device=device, dtype=dtype) for x in d["X_train"]]
                X_test_list = d["X_test"]
                
                Y_train = np.array(d["Y_train"]).flatten().astype(float)
                Y_test = np.array(d["Y_test"]).flatten().astype(float)
                
                has_error = False
                try:
                    projection = get_robust_weights_multi_binary(
                        X=X_train_torch,
                        Y=Y_train,
                        subsampling_percent=[1.0] * len(X),
                        between_discriminate_ratio=BDRatio,
                        lambda_between=lambda_between,
                        lambda_pheno=lambda_pheno,
                        subsampling_num=1,
                        cc_coef=ScalingFactor,
                        ncomp_pls=ncomp_pls,
                        eval_classifier=True,
                        test_data=X_test_list
                    )
                    
                    out_train = projection['out_train']
                    out_test = projection['out_test']
                    
                    train_data = out_train
                    
                    model = sm.GLM(Y_train, train_data, family=sm.families.Binomial())
                    result = model.fit(disp=0)
                    
                    train_pred = result.predict(train_data)
                    test_pred = result.predict(out_test)
                    
                    TrainMetric[idx] = _compute_binary_metrics(Y_train, train_pred, EvalMethod)
                    TestMetric[idx] = _compute_binary_metrics(Y_test, test_pred, EvalMethod)
                    
                except Exception as e:
                    print(f"Caught an error: {e} on iteration {idx}")
                    has_error = True
                
                if has_error:
                    continue
            
            CVResult_df = pd.DataFrame({
                "TrainMetric": TrainMetric,
                "TestMetric": TestMetric
            })
            CVResult_list.append(CVResult_df)
        
        AggregatedCVResult = sum(CVResult_list) / len(CVResult_list)
        
        best_idx = AggregatedCVResult["TestMetric"].idxmax()
        BestPen = PenComb[best_idx, :]
        
        print("\n")
        for xx in range(len(X)):
            print(f"The best penalty term for {DataType[xx]} is: {BestPen[xx]}")
        print(f"The best penalty term on classifier is: {BestPen[len(X)]}")
        
        best_metric = round(AggregatedCVResult.iloc[best_idx]["TestMetric"], 3)
        print(f"Testing {EvalMethod} score: {best_metric}\n")
        
        print("Running multi-omics PLS with best penalty on complete dataset.\n")
        
        # final run: PLS
        outcome = Y.flatten().astype(float)
        X_scaled_final = [torch.tensor(r_scale(x), device=device, dtype=dtype) for x in X]
        
        Ws_final = get_robust_weights_multi_binary(
            X=X_scaled_final,
            Y=outcome,
            subsampling_percent=SubsamplingPercent,
            between_discriminate_ratio=BDRatio,
            lambda_between=BestPen[:len(X)],
            lambda_pheno=BestPen[len(X)],
            subsampling_num=subSampNum,
            cc_coef=ScalingFactor,
            ncomp_pls=ncomp_pls,
            eval_classifier=False
        )
        
        if isinstance(Ws_final, np.ndarray):
            Ws_final = torch.tensor(Ws_final, device=device, dtype=dtype)

    # network construction
    total_features = sum(x.shape[1] for x in X)
    if len(feature_labels) != total_features:
        print(f"Warning: Label mismatch ({len(feature_labels)} labels vs {total_features} features). "
              f"Falling back to generic labels.")
        feature_labels = []
        for i in range(len(X)):
            feature_labels.extend([f"{DataType[i]}_Feat{j+1}" for j in range(X[i].shape[1])])
    
    Abar = get_abar(Ws_final, feature_label=feature_labels)

    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print("\n--------------------------------------------------")
    print(">> Now starting network clustering...")
    print("--------------------------------------------------\n")
    
    modules = get_omics_modules(Abar, cut_height=CutHeight)
    
    print("Clustering completed...\n")
    
    print("--------------------------------------------------")
    print(">> Now starting network pruning and summarization score extraction...")
    print("--------------------------------------------------\n")
    
    # Build combined omics matrix (n x p) for summarization
    X_combined = np.hstack(X)
    
    subnetwork_results = prune_modules(
        Abar=Abar,
        X_combined=X_combined,
        Y=Y,
        modules=modules,
        feature_labels=feature_labels,
        min_size=min_size,
        max_size=max_size,
        summarization=summarization,
        saving_dir=saving_dir
    )


    # save cv results
    PenComb_df = pd.DataFrame(PenComb, columns=[f"l{i+1}" for i in range(PenComb.shape[1])])
    AggregatedCVResultPen = pd.concat([PenComb_df, AggregatedCVResult], axis=1)
    AggregatedCVResultPen.to_csv(os.path.join(saving_dir, "CVResult.csv"), index=False)
    
    # save fold data
    with open(os.path.join(saving_dir, "CVFold.pkl"), 'wb') as f:
        pickle.dump(folddata_cpu, f)
    
    # save global network
    globalNetwork = {"AdjacencyMatrix": Abar, "Data": X, "Phenotype": Y}
    with open(os.path.join(saving_dir, "globalNetwork.pkl"), 'wb') as f:
        pickle.dump(globalNetwork, f)
    
    print("\n************************************")
    print("* Execution Finished!            *")
    print("************************************\n")
    
    return {
        "AdjacencyMatrix": Abar,
        "Data": X,
        "CVResult": AggregatedCVResult,
        "Subnetworks": subnetwork_results
    }


def _is_binary(Y: Union[pd.DataFrame, np.ndarray, list]) -> bool:
    """Check if Y is a binary phenotype (factor-like).

    Args:

        Y (pd.DataFrame | np.ndarray | list): Input phenotype vector to check for binary status.

    Returns:

        bool: True if Y contains exactly two unique values (0 and 1); False otherwise.

    """
    Y_flat = np.array(Y).flatten()
    unique_vals = np.unique(Y_flat[~np.isnan(Y_flat)])
    if len(unique_vals) == 2:
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return True
    return False


def _compute_binary_metrics(y_true: Union[np.ndarray, list], y_pred_prob: Union[np.ndarray, list], eval_method: str) -> float:
    """Compute classification metrics

    Args:

        y_true (np.ndarray | list): True binary labels (0/1) of shape (n,).
        y_pred_prob (np.ndarray | list): Predicted probabilities from logistic regression of shape (n,).
        eval_method (str): Evaluation metric; one of 'accuracy', 'auc', 'precision', 'recall', or 'f1'.

    Returns:

        float: The computed metric value.

    """
    y_true = np.array(y_true).flatten()
    y_pred_prob = np.array(y_pred_prob).flatten()
    
    # AUC calculation
    if eval_method == 'auc':
        desc_order = np.argsort(-y_pred_prob)
        y_sorted = y_true[desc_order]
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tp = 0
        fp = 0
        auc_val = 0.0
        
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
                auc_val += tp / n_pos
        
        auc_val /= n_neg
        return auc_val
    
    # threshold predictions
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    if eval_method == 'accuracy':
        return (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
    
    elif eval_method == 'precision':
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    elif eval_method == 'recall':
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    elif eval_method == 'f1':
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    else:
        raise ValueError(
            f"Invalid eval_method '{eval_method}'. "
            "Choose from: accuracy, auc, precision, recall, f1."
        )