import torch
import numpy as np
import itertools
from typing import List, Union, Optional, Dict, Any
from .math_helpers import binary_search, soft, l2n, r_scale_torch

def my_get_crit(xlist: List[torch.Tensor], ws: List[torch.Tensor], pair_cc: Union[np.ndarray, torch.Tensor], cc_coef: Union[np.ndarray, torch.Tensor]) -> float:
    """PyTorch version of my_get_crit computing SmCCA objective on GPU.

    Args:

        xlist (List[torch.Tensor]): List of input data matrices.
        ws (List[torch.Tensor]): List of weight vectors.
        pair_cc (np.ndarray | torch.Tensor): Matrix of pair indices.
        cc_coef (np.ndarray | torch.Tensor): Vector of scaling coefficients.

    Returns:

        float: Computed objective function value.

    """
    # device management
    device = xlist[0].device

    if not isinstance(cc_coef, torch.Tensor):
        cc_coef = torch.tensor(cc_coef, device=device, dtype=torch.float32)

    if isinstance(pair_cc, torch.Tensor):
        pair_cc = pair_cc.cpu().numpy()

    num_pairs = pair_cc.shape[1]
    crits = []

    # iterate pairs
    for k in range(num_pairs):
        i = int(pair_cc[0, k])
        j = int(pair_cc[1, k])

        proj_i = torch.matmul(xlist[i], ws[i])
        proj_j = torch.matmul(xlist[j], ws[j])

        val = torch.matmul(proj_i.t(), proj_j)

        crits.append(val.view(-1))

    # weighted sum
    if len(crits) > 0:
        crits_vec = torch.cat(crits)
        crit = torch.sum(crits_vec * cc_coef)
    else:
        crit = torch.tensor(0.0, device=device)

    return crit.item()

def my_get_cors(xlist: List[torch.Tensor], ws: List[torch.Tensor], pair_cc: Union[np.ndarray, torch.Tensor], cc_coef: Union[np.ndarray, torch.Tensor]) -> float:
    """PyTorch version of my_get_cors computing total weighted canonical correlations.

    Args:

        xlist (List[torch.Tensor]): List of input data matrices.
        ws (List[torch.Tensor]): List of weight vectors.
        pair_cc (np.ndarray | torch.Tensor): Matrix of pair indices.
        cc_coef (np.ndarray | torch.Tensor): Vector of scaling coefficients.

    Returns:

        float: Total weighted correlation value.

    """
    device = xlist[0].device

    if not isinstance(cc_coef, torch.Tensor):
        cc_coef = torch.tensor(cc_coef, device=device, dtype=torch.float32)

    if isinstance(pair_cc, torch.Tensor):
        pair_cc = pair_cc.cpu().numpy()

    num_pairs = pair_cc.shape[1]
    ccs = []

    for k in range(num_pairs):
        i = int(pair_cc[0, k])
        j = int(pair_cc[1, k])

        # calculate projections
        u = torch.matmul(xlist[i], ws[i]).flatten()
        v = torch.matmul(xlist[j], ws[j]).flatten()

        # pearson correlation
        u_mean = torch.mean(u)
        v_mean = torch.mean(v)

        u_centered = u - u_mean
        v_centered = v - v_mean

        numerator = torch.sum(u_centered * v_centered)

        denom_u = torch.sqrt(torch.sum(u_centered**2))
        denom_v = torch.sqrt(torch.sum(v_centered**2))
        denominator = denom_u * denom_v

        if denominator == 0:
            corr = torch.tensor(0.0, device=device)
        else:
            corr = numerator / denominator

        if torch.isnan(corr):
            corr = torch.tensor(0.0, device=device)

        ccs.append(corr)

    # weighted sum
    if len(ccs) > 0:
        ccs_vec = torch.stack(ccs)
        cors = torch.sum(ccs_vec * cc_coef)
    else:
        cors = torch.tensor(0.0, device=device)

    return cors.item()




def my_update_w(xlist: List[torch.Tensor], i: int, K: int, sumabsthis: float, ws: List[torch.Tensor], ws_final: List[torch.Tensor], pair_cc: Union[np.ndarray, torch.Tensor], cc_coef: Union[np.ndarray, torch.Tensor, List[float]], type: str = "standard") -> torch.Tensor:
    """PyTorch version of my_update_w using hybrid GPU/CPU approach.

    Args:

        xlist (List[torch.Tensor]): List of input data matrices.
        i (int): Index of the current omics layer to update.
        K (int): Number of latent components.
        sumabsthis (float): Sparsity penalty (L1 sum constraint) for this layer.
        ws (List[torch.Tensor]): Current weight vectors.
        ws_final (List[torch.Tensor]): Final weight vectors from previous components.
        pair_cc (np.ndarray | torch.Tensor): Matrix of pair indices.
        cc_coef (np.ndarray | torch.Tensor | List[float]): Scaling coefficients.
        type (str): Analysis type; currently supports 'standard'.

    Returns:

        torch.Tensor: Updated weight vector for the i-th layer.

    """
    device = xlist[i].device
    dtype = xlist[i].dtype

    if isinstance(cc_coef, torch.Tensor):
        cc_coef_list = cc_coef.cpu().tolist()
    elif isinstance(cc_coef, np.ndarray):
        cc_coef_list = cc_coef.tolist()
    else:
        cc_coef_list = list(cc_coef)

    if isinstance(pair_cc, torch.Tensor):
        pair_cc = pair_cc.cpu().numpy()
    else:
        pair_cc = np.array(pair_cc)

    tots = 0
    num_pairs = len(cc_coef_list)

    # phase 1: matrix operations (gpu)
    for x in range(num_pairs):
        pairx = pair_cc[:, x]

        if pairx[0] != i and pairx[1] != i:
            continue
        else:
            if pairx[0] == i:
                j = int(pairx[1])
            elif pairx[1] == i:
                j = int(pairx[0])

            Xi = xlist[i]
            Xj = xlist[j]

            diagmat = torch.matmul(
                torch.matmul(ws_final[i].t(), Xi.t()),
                torch.matmul(Xj, ws_final[j])
            )

            diagmat = torch.diag(torch.diag(diagmat))

            term1 = torch.matmul(Xi.t(), torch.matmul(Xj, ws[j]))

            term2_inner = torch.matmul(diagmat, torch.matmul(ws_final[j].t(), ws[j]))
            term2 = torch.matmul(ws_final[i], term2_inner)

            y = term1 - term2
            y = y * cc_coef_list[x]

            tots = tots + y

    # phase 2: scalar optimization (cpu)
    if type == "standard":
        tots_cpu = tots.cpu().numpy()

        sumabsthis = binary_search(tots_cpu, sumabsthis)
        numerator = soft(tots_cpu, sumabsthis)
        denominator = l2n(numerator)
        w_cpu = numerator / denominator

        w = torch.tensor(w_cpu, device=device, dtype=dtype)

    else:
        raise ValueError("Current version requires all element types to be standard (not ordered).")

    return w

def my_multi_cca(xlist: List[torch.Tensor], penalty: Optional[Union[float, List[float], np.ndarray]] = None, ws: Optional[List[torch.Tensor]] = None, niter: int = 25, type: str = "standard", ncomponents: int = 1, trace: bool = True, standardize: bool = True, cc_coef: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None) -> Dict[str, Any]:
    """PyTorch version of my_multi_cca performing sparse multiple canonical correlation analysis (SmCCA) on GPU.

    Args:

        xlist (List[torch.Tensor]): List of data matrices (torch tensors on the target device).
        penalty (float | List[float] | np.ndarray | None): Penalty parameters for each omics layer.
        ws (List[torch.Tensor] | None): Initial weight vectors; if None, initialized via SVD.
        niter (int): Maximum number of iterations for convergence.
        type (str): Analysis type; currently supports "standard".
        ncomponents (int): Number of canonical components to extract.
        trace (bool): If True, print iteration progress.
        standardize (bool): If True, standardize input data before analysis.
        cc_coef (List | np.ndarray | torch.Tensor | None): Scaling coefficients for between-omics pairs.

    Returns:

        Dict[str, Any]: Dictionary containing weights ('ws'), initialization ('ws_init'), and correlations ('cors').

    """
    # device management
    device = xlist[0].device
    dtype = xlist[0].dtype

    last_dims = xlist[-1].shape
    last_ncol = last_dims[1] if len(last_dims) > 1 else 1

    # branch 1: standard multicca
    if last_ncol > 1:
        K = len(xlist)
        pairs = list(itertools.combinations(range(K), 2))
        pair_cc = np.array(pairs).T
        num_cc = pair_cc.shape[1]

        if cc_coef is None:
            cc_coef = torch.ones(num_cc, device=device, dtype=dtype)
        else:
            if not isinstance(cc_coef, torch.Tensor):
                cc_coef = torch.tensor(cc_coef, device=device, dtype=dtype)

        if cc_coef.numel() != num_cc:
            raise ValueError(f"Invalid coefficients. Provide {num_cc} values.")

        if isinstance(type, str):
            if type != "standard":
                raise ValueError("Phenotype data must be continuous/standard.")
            type_vec = np.array([type] * K)
        else:
            type_vec = np.array(type)

        if len(type_vec) != K:
            raise ValueError("Type must be vector of length 1 or K.")

        if standardize:
            xlist = [r_scale_torch(x) for x in xlist]

        if ws is not None:
            make_null = False
            for i in range(K):
                if ws[i].shape[1] < ncomponents:
                    make_null = True
            if make_null:
                ws = None

        if ws is None:
            ws = []
            for i in range(K):
                U, S, Vh = torch.linalg.svd(xlist[i], full_matrices=False)
                V = Vh.T
                ws.append(V[:, :ncomponents])

        ws_init = [w.clone() for w in ws]

        if penalty is None:
            penalty = np.full(K, np.nan)
            for k in range(K):
                if type_vec[k] == "standard":
                    penalty[k] = 4

        if np.ndim(penalty) == 0:
            penalty = np.full(K, penalty)
        else:
            penalty = np.array(penalty)

        ws_final = [w.clone() for w in ws_init]
        for i in range(K):
            ws_final[i] = torch.zeros((xlist[i].shape[1], ncomponents), device=device, dtype=dtype)

        cors = []

        # optimization loop
        for comp in range(ncomponents):
            ws_curr = []
            for i in range(K):
                ws_curr.append(ws_init[i][:, comp].reshape(-1, 1))

            curiter = 1
            crit_old = -10.0
            crit = -20.0
            storecrits = []

            while (curiter <= niter and abs(crit_old - crit) / abs(crit_old) > 0.001 and crit_old != 0):

                crit_old = crit
                crit = my_get_crit(xlist, ws_curr, pair_cc, cc_coef)
                storecrits.append(crit)

                if trace:
                    print(curiter, end=" ", flush=True)
                curiter += 1

                for i in range(K):
                    ws_curr[i] = my_update_w(xlist, i, K, penalty[i], ws_curr, ws_final, pair_cc, cc_coef, type=type_vec[i])

            if trace:
                print("")

            for i in range(K):
                ws_final[i][:, comp] = ws_curr[i].flatten()

            cors.append(my_get_cors(xlist, ws_curr, pair_cc, cc_coef))

        return {
            "ws": ws_final,
            "ws_init": ws_init,
            "K": K,
            "type": type_vec,
            "penalty": penalty,
            "cors": cors
        }

    else:
        # branch 2: phenotype included
        K = len(xlist)
        pairs = list(itertools.combinations(range(K), 2))
        pair_cc = np.array(pairs).T
        num_cc = pair_cc.shape[1]

        if cc_coef is None:
            cc_coef = torch.ones(num_cc, device=device, dtype=dtype)
        else:
            if not isinstance(cc_coef, torch.Tensor):
                cc_coef = torch.tensor(cc_coef, device=device, dtype=dtype)

        if isinstance(type, str):
            if type != "standard":
                raise ValueError("Phenotype data must be continuous/standard.")
            type_vec = np.array([type] * K)
        else:
            type_vec = np.array(type)

        if standardize:
            xlist = [r_scale_torch(x) for x in xlist]

        if ws is not None:
            make_null = False
            for i in range(K - 1):
                if ws[i].shape[1] < ncomponents:
                    make_null = True
            if make_null:
                ws = None

        if ws is None:
            ws = []
            for i in range(K - 1):
                U, S, Vh = torch.linalg.svd(xlist[i], full_matrices=False)
                V = Vh.T
                ws.append(V[:, :ncomponents])
            # phenotype weight fixed at 1.0
            ws.append(torch.tensor([[1.0]], device=device, dtype=dtype))

        ws_init = [w.clone() for w in ws]

        if penalty is None:
            penalty = np.full(K, np.nan)
            for k in range(K):
                if type_vec[k] == "standard":
                    penalty[k] = 4

        if np.ndim(penalty) == 0:
            penalty = np.full(K, penalty)
        else:
            penalty = np.array(penalty)

        ws_final = [w.clone() for w in ws_init]
        for i in range(K - 1):
            ws_final[i] = torch.zeros((xlist[i].shape[1], ncomponents), device=device, dtype=dtype)

        cors = []

        # optimization loop
        for comp in range(ncomponents):
            ws_curr = []
            for i in range(K - 1):
                ws_curr.append(ws_init[i][:, comp].reshape(-1, 1))
            ws_curr.append(torch.tensor([[1.0]], device=device, dtype=dtype))

            curiter = 1
            crit_old = -10.0
            crit = -20.0
            storecrits = []

            while (curiter <= niter and abs(crit_old - crit) / abs(crit_old) > 0.001 and crit_old != 0):

                crit_old = crit
                crit = my_get_crit(xlist, ws_curr, pair_cc, cc_coef)
                storecrits.append(crit)

                if trace:
                    print(curiter, end=" ", flush=True)
                curiter += 1

                # update k-1 datasets
                for i in range(K - 1):
                    ws_curr[i] = my_update_w(xlist, i, K, penalty[i], ws_curr, ws_final, pair_cc, cc_coef, type=type_vec[i])

            if trace:
                print("")

            for i in range(K - 1):
                ws_final[i][:, comp] = ws_curr[i].flatten()

            cors.append(my_get_cors(xlist, ws_curr, pair_cc, cc_coef))

        return {
            "ws": ws_final,
            "ws_init": ws_init,
            "K": K,
            "type": type_vec,
            "penalty": penalty,
            "cors": cors
        }
