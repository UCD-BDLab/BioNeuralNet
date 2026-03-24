import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Optional, Dict

# internal imports
from .math_helpers import r_scale_torch, _splsda, r_scale
from .core import my_multi_cca
import statsmodels.api as sm

def get_can_weights_multi(X: List[torch.Tensor], Trait: Optional[torch.Tensor] = None, Lambda: Optional[Union[List[float], np.ndarray]] = None, cc_coef: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None, no_trait: bool = True, trace: bool = False, trait_weight: bool = False) -> List[torch.Tensor]:
    """PyTorch version of get_can_weights_multi wrapper.

    Args:

        X (List[torch.Tensor]): List of input data matrices on target device.
        Trait (torch.Tensor | None): Optional trait data tensor.
        Lambda (List[float] | np.ndarray | None): Penalty parameters; required.
        cc_coef (List | np.ndarray | torch.Tensor | None): Pairwise correlation coefficients.
        no_trait (bool): If True, run unsupervised CCA; if False, include trait.
        trace (bool): If True, print trace info during optimization.
        trait_weight (bool): If True, return trait weights in output list.

    Returns:

        List[torch.Tensor]: List of weight tensors for each input matrix.

    """

    # input validation
    if Lambda is None:
        raise ValueError("Lambda must be provided.")
    Lambda = np.atleast_1d(np.array(Lambda, dtype=float))
    
    for lam in Lambda:
        if abs(lam - 0.5) > 0.5:
            raise ValueError("Invalid penalty parameter. Lambda1 needs to be between zero and one.")
    
    if np.min(Lambda) == 0:
        raise ValueError("Invalid penalty parameter. Both Lambda1 and Lambda2 has to be greater than 0.")

    # penalty calculation
    current_X = list(X)
    L = []
    for i in range(len(current_X)):
        ncol = current_X[i].shape[1]
        val = max(1, np.sqrt(ncol) * Lambda[i])
        L.append(val)

    # cca execution
    if no_trait:
        out = my_multi_cca(current_X, penalty=L, cc_coef=cc_coef, trace=trace)
    else:
        if Trait is None:
            raise ValueError("Trait must be provided if no_trait is False.")
        
        scaled_trait = r_scale_torch(Trait)
        current_X.append(scaled_trait)
        
        trait_ncol = scaled_trait.shape[1]
        L.append(np.sqrt(trait_ncol))
        
        out = my_multi_cca(current_X, penalty=L, cc_coef=cc_coef, trace=trace)

    # output extraction
    if trait_weight:
        ws = out['ws']
    else:
        if no_trait:
            ws = out['ws']
        else:
            ws = out['ws'][:-1]
            
    return ws
def get_robust_weights_multi(X: List[torch.Tensor], Trait: Optional[torch.Tensor], Lambda: Union[List[float], np.ndarray], s: Optional[Union[List[float], np.ndarray]] = None, no_trait: bool = False, subsampling_num: int = 1000, cc_coef: Optional[np.ndarray] = None, trace: bool = False, trait_weight: bool = False) -> torch.Tensor:
    """PyTorch version of get_robust_weights_multi with subsampling loop.

    Args:

        X (List[torch.Tensor]): List of input data matrices on target device.
        Trait (torch.Tensor | None): Trait data tensor or None.
        Lambda (List[float] | np.ndarray): Penalty parameters for CCA/PLS.
        s (List[float] | np.ndarray | None): Subsampling proportions for each omics layer.
        no_trait (bool): If True, compute weights without using Trait information (unsupervised).
        subsampling_num (int): Number of subsampling iterations to perform.
        cc_coef (np.ndarray | None): Scaling coefficients for between-omics relationships.
        trace (bool): If True, print trace information during execution.
        trait_weight (bool): If True, include trait weights in the output.

    Returns:

        torch.Tensor: Matrix of weights with shape (total_features, subsampling_num) on the same device as X.

    """
    # device management
    device = X[0].device
    dtype = X[0].dtype
    
    if s is None:
        raise ValueError("s (subsampling proportions) must be provided.")
    
    s = np.atleast_1d(np.array(s, dtype=float))
    if s.size == 1 and len(X) > 1:
        s = np.repeat(s, len(X))
        
    Lambda = np.atleast_1d(np.array(Lambda, dtype=float))

    # validation checks
    if np.sum(s == 0) > 1:
        raise ValueError("Subsampling proportion needs to be greater than zero.")
    else:
        if np.sum(np.abs(s - 0.5) > 0.5) > 0:
            raise ValueError("Subsampling proportions can not exceed one.")
            
    if (np.sum(np.abs(Lambda - 0.5) > 0.5) > 0) or (np.sum(Lambda == 0) > 0):
        raise ValueError("Invalid penalty parameter. Lambda1 needs to be between zero and one.")

    # setup dimensions
    p_data = np.array([x.shape[1] for x in X])
    p = int(np.sum(p_data))
    p_sub = np.ceil(p_data * s).astype(int)

    # subsampling loop
    results = []
    
    iter_range = range(subsampling_num)
    if subsampling_num > 1:
        iter_range = tqdm(range(subsampling_num), desc="Robust Weights")

    for _ in iter_range:
        # sampling
        samp = []
        for h in range(len(p_data)):
            indices = np.random.choice(p_data[h], p_sub[h], replace=False)
            indices.sort()
            samp.append(indices)

        # subset and scale
        x_par = []
        for h in range(len(p_data)):
            subset = X[h][:, samp[h]]
            x_par.append(r_scale_torch(subset))

        # compute weights
        if Trait is not None:
            out = get_can_weights_multi(x_par, Trait, Lambda, no_trait=no_trait, trace=trace, cc_coef=cc_coef)
        else:
            out = get_can_weights_multi(x_par, None, Lambda, no_trait=True, trace=trace, cc_coef=cc_coef)

        # reconstruct weight vector
        w = torch.zeros(p, device=device, dtype=dtype)
        p_cum = np.insert(np.cumsum(p_data), 0, 0)

        for h in range(len(p_cum) - 1):
            global_indices = samp[h] + int(p_cum[h])
            idx = torch.tensor(global_indices, device=device, dtype=torch.long)
            
            val = out[h]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(np.array(val).flatten(), device=device, dtype=dtype)
            else:
                val = val.flatten()
                
            w[idx] = val

        results.append(w)

    # stack results
    beta = torch.stack(results, dim=1)
    return beta

def get_robust_weights_single_binary(X1: np.ndarray, Trait: np.ndarray, Lambda1: float, s1: float = 0.7, subsampling_num: int = 1000, K: int = 3) -> np.ndarray:
    """Compute aggregated sparse PLS-DA canonical weights for single omics data with binary phenotype.

    Args:

        X1 (np.ndarray): Input data matrix of shape (n, p1).
        Trait (np.ndarray): Binary phenotype vector (0/1) of shape (n,).
        Lambda1 (float): LASSO penalty parameter for SPLSDA; between 0 and 1.
        s1 (float): Proportion of features to subsample per iteration.
        subsampling_num (int): Number of subsampling iterations.
        K (int): Number of latent components for PLS-DA.

    Returns:

        np.ndarray: Weight matrix of shape (p1, subsampling_num).

    """
    X1 = np.array(X1, dtype=float)
    Trait = np.array(Trait).flatten()
    
    p1 = X1.shape[1]
    p1_sub = int(np.ceil(s1 * p1))
    
    results = []
    
    iter_range = range(subsampling_num)
    if subsampling_num > 1:
        iter_range = tqdm(range(subsampling_num), desc="Single Binary Weights")
    
    for _ in iter_range:
        # subsample features
        samp1 = np.sort(np.random.choice(p1, p1_sub, replace=False))
        
        # scale subsampled data
        x1_par = r_scale(X1[:, samp1])
        
        # run sparse pls-da
        out = _splsda(x=x1_par, y=Trait, K=K, eta=Lambda1, kappa=0.5, scale_x=False)
        
        u = np.zeros(p1_sub)
        w = np.zeros(p1)
        
        T_scores = out['T']
        W_weights = out['W']
        A_indices = out['A']
        
        # fit logistic regression on latent factors
        try:
            model = sm.GLM(Trait, T_scores, family=sm.families.Binomial())
            result = model.fit(disp=0)
            glm_coefs = result.params
        except Exception:
            results.append(np.zeros(p1))
            continue
        
        # compute weights
        u[A_indices] = np.abs(W_weights) @ np.abs(glm_coefs)
        
        # normalize
        norm_val = np.linalg.norm(u[A_indices])
        if norm_val > 0:
            u[A_indices] = u[A_indices] / norm_val
        
        # scatter back to full feature space
        w[samp1] = u
        results.append(w)
    
    beta = np.column_stack(results)
    return beta

def get_robust_weights_multi_binary(X: List[torch.Tensor], Y: Union[np.ndarray, torch.Tensor], between_discriminate_ratio: Optional[Union[List[float], np.ndarray]] = None, subsampling_percent: Optional[Union[List[float], np.ndarray]] = None, cc_coef: Optional[np.ndarray] = None, lambda_between: Optional[Union[List[float], np.ndarray]] = None, lambda_pheno: Optional[float] = None, subsampling_num: int = 1000, ncomp_pls: int = 3, eval_classifier: bool = False, test_data: Optional[List[Union[np.ndarray, torch.Tensor]]] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """PyTorch version of get_robust_weights_multi_binary using hybrid GPU/CPU execution.

    Args:

        X (List[torch.Tensor]): List of omics data matrices (torch tensors on target device).
        Y (np.ndarray | torch.Tensor): Binary phenotype vector.
        between_discriminate_ratio (List[float] | np.ndarray | None): Ratio for weighting between-omics vs omics-phenotype contributions.
        subsampling_percent (List[float] | np.ndarray | None): Proportion of features to subsample per iteration.
        cc_coef (np.ndarray | None): Pairwise correlation coefficients.
        lambda_between (List[float] | np.ndarray | None): Penalty terms for between-omics CCA.
        lambda_pheno (float | None): Penalty term for omics-phenotype PLS.
        subsampling_num (int): Number of subsampling iterations.
        ncomp_pls (int): Number of latent components for PLS.
        eval_classifier (bool): If True, return projections for classifier evaluation instead of weights.
        test_data (List[np.ndarray | torch.Tensor] | None): Test data matrices required if eval_classifier is True.

    Returns:

        np.ndarray | Dict: Weight matrix (if eval_classifier=False) or dictionary containing train/test projections.

    """
    if between_discriminate_ratio is None:
        between_discriminate_ratio = [1, 1]
    between_discriminate_ratio = np.array(between_discriminate_ratio, dtype=float)
    
    if lambda_between is None:
        raise ValueError("lambda_between must be provided.")
    lambda_between = np.array(lambda_between)
    
    eta = lambda_pheno
    
    # ensure Y is numpy
    if isinstance(Y, torch.Tensor):
        Y_np = Y.cpu().numpy().flatten()
    else:
        Y_np = np.array(Y).flatten()

    # step 1: between-omics smcca (gpu)
    between_omics_weight = get_robust_weights_multi(
        X, Trait=None, Lambda=lambda_between, s=subsampling_percent,
        no_trait=True, cc_coef=cc_coef, subsampling_num=subsampling_num
    )
    
    # move to cpu for pls-da
    between_omics_weight = between_omics_weight.cpu().numpy()
    
    # column-bind all omics
    X_all = np.hstack([x.cpu().numpy() for x in X])
    
    # feature type index
    type_index = np.concatenate([np.full(X[h].shape[1], h) for h in range(len(X))])
    
    if not eval_classifier:
        # branch a: network construction
        n_subsamples = between_omics_weight.shape[1]
        p_total = between_omics_weight.shape[0]
        
        omics_pheno_weight = np.zeros_like(between_omics_weight)
        
        for iii in tqdm(range(n_subsamples), desc="Omics-Phenotype PLS"):
            selected_mask = between_omics_weight[:, iii] != 0
            selected_indices = np.where(selected_mask)[0]
            
            if len(selected_indices) == 0:
                continue
            
            X_subset = X_all[:, selected_indices]
            
            try:
                # cpu: small matrix pls-da
                Ws_pheno = get_robust_weights_single_binary(
                    X1=X_subset, Trait=Y_np.reshape(-1, 1),
                    Lambda1=float(eta), s1=1.0,
                    subsampling_num=1, K=ncomp_pls
                )
            except Exception:
                continue
            
            omics_pheno_weight[selected_indices, iii] = Ws_pheno.flatten()
            
            # normalize per data type
            for j in range(len(X)):
                type_mask = type_index == j
                norm_val = np.linalg.norm(omics_pheno_weight[type_mask, iii])
                if norm_val > 0:
                    omics_pheno_weight[type_mask, iii] /= norm_val
        
        # zero out between-omics where pheno is zero
        between_omics_weight[omics_pheno_weight == 0] = 0
        
        # remove zero/nan columns
        if subsampling_num > 1:
            zero_cols = []
            for col_idx in range(omics_pheno_weight.shape[1]):
                col = omics_pheno_weight[:, col_idx]
                if np.all(col == 0) or np.any(np.isnan(col)):
                    zero_cols.append(col_idx)
            
            if len(zero_cols) > 0:
                keep_cols = np.setdiff1d(np.arange(omics_pheno_weight.shape[1]), zero_cols)
                between_omics_weight = between_omics_weight[:, keep_cols]
                omics_pheno_weight = omics_pheno_weight[:, keep_cols]
        
        # aggregate
        ratio_sum = np.sum(between_discriminate_ratio)
        w1 = between_discriminate_ratio[0] / ratio_sum
        w2 = between_discriminate_ratio[1] / ratio_sum
        
        cc_weight = w1 * between_omics_weight + w2 * omics_pheno_weight
        
        return cc_weight
    
    else:
        # branch b: classifier evaluation
        if subsampling_num != 1:
            raise ValueError("Subsampling number must be 1 when evaluating the classifier.")
        
        if test_data is None:
            raise ValueError("test_data must be provided when eval_classifier=True.")
        
        selected_mask = between_omics_weight[:, 0] != 0
        selected_indices = np.where(selected_mask)[0]
        
        X_subset = X_all[:, selected_indices]
        
        try:
            out = _splsda(x=r_scale(X_subset), y=Y_np, K=ncomp_pls, eta=lambda_pheno, kappa=0.5, scale_x=False)
            
            out_data = np.zeros((X_subset.shape[1], ncomp_pls))
            out_data[out['A'], :] = out['W']
            
            # process test data
            X_all_test = np.hstack([t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in test_data])
            X_subset_test = X_all_test[:, selected_indices]
            out_test = X_subset_test @ out_data
            
            out_train = out['T']
            
            return {'out_train': out_train, 'out_test': out_test}
            
        except Exception as e:
            print(f"Caught an error: {e}")
            n_train = X_all.shape[0]
            n_test = np.hstack([t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in test_data]).shape[0]
            out_train = np.zeros((n_train, ncomp_pls))
            out_test = np.zeros((n_test, ncomp_pls))
            return {'out_train': out_train, 'out_test': out_test}