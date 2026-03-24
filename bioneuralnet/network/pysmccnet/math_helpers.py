import numpy as np
import torch
import warnings
from typing import Union, List, Dict, Any

def l2n(vec: Union[np.ndarray, list]) -> float:
    """Computes the L2 norm of a vector; returns 0.05 if norm is zero.

    Args:

        vec (np.ndarray | list): Input vector.

    Returns:

        float: The L2 norm or 0.05 if zero.

    """
    vec = np.array(vec)
    a = np.sqrt(np.sum(vec**2))
    if a == 0:
        a = 0.05
    return a


def soft(x: Union[np.ndarray, list, float], d: float) -> Union[np.ndarray, float]:
    """Applies soft thresholding to input x with threshold d.

    Args:

        x (np.ndarray | list | float): Input data.
        d (float): Threshold value.

    Returns:

        np.ndarray | float: Thresholded output; sign(x) * max(0, abs(x) - d).

    """
    x = np.array(x)
    return np.sign(x) * np.maximum(0, np.abs(x) - d)


def binary_search(argu: Union[np.ndarray, list], sumabs: float) -> float:
    """Finds penalty parameter such that L1 norm of normalized argument equals sumabs.

    Args:

        argu (np.ndarray | list): Input vector.
        sumabs (float): Target L1 sum constraint.

    Returns:

        float: The computed penalty parameter (lambda).

    """
    argu = np.array(argu)
    
    # calculate norm
    norm_val = l2n(argu)
    
    # check conditions
    if norm_val == 0 or np.sum(np.abs(argu / norm_val)) <= sumabs:
        return 0

    lam1 = 0
    lam2 = np.max(np.abs(argu)) - 1e-5
    iter_count = 1
    
    while iter_count < 150:
        mid_val = (lam1 + lam2) / 2
        su = soft(argu, mid_val)
        
        su_norm = l2n(su)
        
        if np.sum(np.abs(su / su_norm)) < sumabs:
            lam2 = mid_val
        else:
            lam1 = mid_val
            
        if (lam2 - lam1) < 1e-6:
            return (lam1 + lam2) / 2
            
        iter_count += 1

    warnings.warn("Didn't quite converge")
    return (lam1 + lam2) / 2


def r_vec_mult_sum(v1: Union[torch.Tensor, np.ndarray, list], v2: Union[torch.Tensor, np.ndarray, list]) -> float:
    """Computes element-wise multiplication and sum with vector recycling.

    Args:

        v1 (torch.Tensor | np.ndarray | list): First input vector.
        v2 (torch.Tensor | np.ndarray | list): Second input vector.

    Returns:

        float: Sum of element-wise product after recycling to matching lengths.

    """
    # ensure 1d tensors
    if not isinstance(v1, torch.Tensor):
        v1 = torch.tensor(np.array(v1).flatten(), dtype=torch.float64)
    else:
        v1 = v1.flatten()
        
    if not isinstance(v2, torch.Tensor):
        v2 = torch.tensor(np.array(v2).flatten(), dtype=torch.float64)
    else:
        v2 = v2.flatten()
    
    # device alignment
    device = v1.device
    v2 = v2.to(device)
    
    n1 = v1.numel()
    n2 = v2.numel()
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    target_len = max(n1, n2)
    
    # recycling via repeat
    if n1 < target_len:
        repeats = (target_len // n1) + 1
        v1 = v1.repeat(repeats)[:target_len]
        
    if n2 < target_len:
        repeats = (target_len // n2) + 1
        v2 = v2.repeat(repeats)[:target_len]
        
    return torch.sum(v1 * v2).item()


def r_scale_torch(x: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
    """Pytorch scaling using sample standard deviation.

    Args:

        x (torch.Tensor | np.ndarray | list): Input data; converted to float32 tensor if not already.

    Returns:

        torch.Tensor: Centered and scaled data tensor.

    """
    # convert to tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # handle 1d arrays
    if x.ndim == 1:
        x = x.view(-1, 1)
        
    # calculate mean
    mean = torch.mean(x, dim=0)
    
    # calculate std
    std = torch.std(x, dim=0, unbiased=True)
    
    # handle constant columns
    std[std == 0] = 1.0
    
    return (x - mean) / std


def r_scale(x: Union[np.ndarray, list]) -> np.ndarray:
    """Numpy scaling using sample standard deviation.

    Args:

        x (np.ndarray | list): Input data matrix.

    Returns:

        np.ndarray: Scaled data where each column has mean 0 and sample std 1.

    """
    # convert to float numpy
    x = np.array(x, dtype=float)
    
    # handle 1d arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # calculate mean
    mean = np.mean(x, axis=0)
    
    # calculate std
    std = np.std(x, axis=0, ddof=1) 
    
    # handle constant columns
    std[std == 0] = 1.0
    
    return (x - mean) / std

def _splsda(x: np.ndarray, y: Union[np.ndarray, list], K: int = 3, eta: float = 0.5, kappa: float = 0.5, scale_x: bool = False) -> Dict[str, Union[np.ndarray, List[int]]]:
    """Sparse PLS-DA implementation

    Args:

        x (np.ndarray): Predictor matrix of shape (n, p).
        y (np.ndarray | list): Binary response vector (0/1) of shape (n,).
        K (int): Number of latent components.
        eta (float): Sparsity parameter (0 to 1); higher values induce more sparsity.
        kappa (float): Ridge mixing parameter for direction estimation.
        scale_x (bool): If True, scale x internally before processing.

    Returns:

        Dict: Dictionary containing 'T' (scores), 'W' (weights), and 'A' (selected indices).

    """
    n, p = x.shape
    y_vec = np.array(y).flatten().astype(float)
    
    if scale_x:
        x = r_scale(x)
    
    X_res = x.copy()
    T_scores = np.zeros((n, K))
    W_full = np.zeros((p, K))
    
    K_actual = min(K, min(n, p))
    
    for k in range(K_actual):
        # direction vector
        if kappa > 0:
            XtX = X_res.T @ X_res
            ridge = kappa * np.eye(p)
            Xty = X_res.T @ y_vec
            try:
                w = np.linalg.solve(XtX + ridge, Xty)
            except np.linalg.LinAlgError:
                w = X_res.T @ y_vec
        else:
            w = X_res.T @ y_vec
        
        # normalize
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w / w_norm
        
        # soft thresholding
        threshold = eta * np.max(np.abs(w))
        w_sparse = np.sign(w) * np.maximum(0, np.abs(w) - threshold)
        
        # normalize sparse weights
        w_sparse_norm = np.linalg.norm(w_sparse)
        if w_sparse_norm > 0:
            w_sparse = w_sparse / w_sparse_norm
        
        # compute scores
        t = X_res @ w_sparse
        t_norm_sq = t @ t
        
        if t_norm_sq == 0:
            T_scores[:, k] = 0
            W_full[:, k] = 0
            continue
        
        # deflation
        p_loading = X_res.T @ t / t_norm_sq
        X_res = X_res - np.outer(t, p_loading)
        
        T_scores[:, k] = t
        W_full[:, k] = w_sparse
    
    # identify selected variables
    selected_mask = np.any(W_full != 0, axis=1)
    A = np.where(selected_mask)[0]
    
    # fallback for empty selection
    if len(A) == 0:
        importance = np.sum(np.abs(W_full), axis=1)
        n_keep = max(1, int(p * (1 - eta)))
        A = np.argsort(importance)[-n_keep:]
        A = np.sort(A)
    
    W_selected = W_full[A, :]
    
    return {
        'T': T_scores,
        'W': W_selected,
        'A': A
    }