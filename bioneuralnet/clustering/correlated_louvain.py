"""
Correlated Louvain Community Detection.

This module extends the standard Louvain algorithm by incorporating an 
absolute phenotype-correlation objective into the modularity maximization 
process.

References:
    Abdel-Hafiz et al. (2022), "Significant Subgraph Detection in 
    Multi-omics Networks for Disease Pathway Identification," 
    Frontiers in Big Data.

Notes:
    **Hybrid Modularity Objective**
    The algorithm optimizes connectivity and phenotype correlation 
    simultaneously using the following weighted objective function:

    .. math::
        Q_{hybrid} = k_L Q + (1 - k_L) \rho

    Where:
        * :math:`Q`: Standard modularity (internal connectivity).
        * :math:`\\rho`: Absolute Pearson correlation of the community's 
          first principal component (PC1) with phenotype :math:`Y`.
        * :math:`k_L`: User-defined weight on modularity (Suggested: 0.2).

Algorithm:
    The hierarchical loop and Phase 2 (network aggregation) remain 
    identical to the standard Louvain method. The modification occurs 
    exclusively in **Phase 1 (Local Optimization)**.

    When evaluating the movement of node :math:`v` from community :math:`D` 
    to community :math:`C`, the gain is calculated as:

    .. math::
        \Delta_{hybrid} = k_L \Delta Q + (1 - k_L) \Delta \\rho

    The correlation gain :math:`\Delta \\rho` is defined as the change in 
    total correlation across affected communities:

    .. math::
        \Delta \\rho = [|\\rho(D \setminus \{v\})| + |\\rho(C \cup \{v\})|] - [|\\rho(D)| + |\\rho(C)|]
"""

import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx

from .louvain import Louvain

logger = logging.getLogger(__name__)

class CorrelatedLouvain(Louvain):
    """
    Correlated Louvain community detection.

    Inherits from :class:`Louvain`.

    Args:

        G (nx.Graph): The input graph for community detection.
        B (pd.DataFrame): Omics data (n_samples x n_features). Column names must match nodes.
        Y (Union[pd.Series, pd.DataFrame]): Phenotype vector aligned with rows of B.
        k_L (float): Weight on modularity in combined objective (Eq. 9).
        weight (str): Edge attribute name for weights.
        max_passes (int): Maximum number of passes for Phase 1 optimization.
        min_delta (float): Convergence tolerance for objective gain.
        seed (Optional[int]): Random seed for reproducibility.
    """

    def __init__(
        self,
        G: nx.Graph,
        B: pd.DataFrame,
        Y: Union[pd.Series, pd.DataFrame],
        k_L: float = 0.2,
        weight: str = "weight",
        max_passes: int = 50,
        min_delta: float = 1e-6,
        seed: Optional[int] = None,
    ):
        if not 0.0 <= k_L <= 1.0:
            raise ValueError(f"k_L must be in [0, 1], got {k_L}")

        super().__init__(
            G=G, 
            weight=weight, 
            max_passes=max_passes,
            min_delta=min_delta, 
            seed=seed,
        )

        self.k_L = k_L
        self.valid_mask = np.zeros(self.n, dtype=bool)
        cols = []

        for idx, node in enumerate(self.nodes):
            col_name = str(node)
            if col_name in B.columns:
                cols.append(B[col_name].values.astype(np.float64))
                self.valid_mask[idx] = True
            else:
                cols.append(np.zeros(len(B), dtype=np.float64))

        omics = np.column_stack(cols)
        mu = omics.mean(axis=0, keepdims=True)
        sigma = omics.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        self.omics_z = (omics - mu) / sigma

        if isinstance(Y, pd.DataFrame):
            y_np = Y.iloc[:, 0].values.astype(np.float64)
        elif isinstance(Y, pd.Series):
            y_np = Y.values.astype(np.float64)
        else:
            y_np = np.asarray(Y, dtype=np.float64)

        self.y_c = y_np - y_np.mean()
        self.y_ss = float((self.y_c ** 2).sum())
        self._corr_cache: Dict[FrozenSet[int], float] = {}

        logger.info(
            f"CorrelatedLouvain: n={self.n}, k_L={k_L}, "
            f"matched_features={int(self.valid_mask.sum())}/{self.n}"
        )

    def _pc1_correlation(self, orig_indices: FrozenSet[int]) -> float:
        """
        Compute |Pearson(PC1(omics[:, indices]), Y)| with caching.

        Args:

            orig_indices (FrozenSet[int]): Set of original node indices.

        Returns:

            float: The absolute correlation value.
        """
        if orig_indices in self._corr_cache:
            return self._corr_cache[orig_indices]

        if len(orig_indices) < 2:
            self._corr_cache[orig_indices] = 0.0
            return 0.0

        idx_arr = np.array(sorted(orig_indices), dtype=np.intp)
        valid_idx = idx_arr[self.valid_mask[idx_arr]]

        if len(valid_idx) < 2:
            self._corr_cache[orig_indices] = 0.0
            return 0.0

        sub = self.omics_z[:, valid_idx]

        try:
            U, S, _ = np.linalg.svd(sub, full_matrices=False)
            pc1 = U[:, 0] * S[0]
            pc1_c = pc1 - pc1.mean()
            pc1_ss = (pc1_c ** 2).sum()

            if pc1_ss < 1e-10:
                val = 0.0
            else:
                val = abs(float(
                    (pc1_c * self.y_c).sum()
                    / np.sqrt(pc1_ss * self.y_ss + 1e-12)
                ))
        except Exception:
            val = 0.0

        self._corr_cache[orig_indices] = val
        return val

    @staticmethod
    def _collect_orig(
        comm_id: int,
        community: np.ndarray,
        n2o: Dict[int, Set[int]],
    ) -> FrozenSet[int]:
        """
        Collect all original-node indices belonging to a specific community.

        Args:

            comm_id (int): The ID of the community.
            community (np.ndarray): The current community assignments.
            n2o (Dict[int, Set[int]]): Mapping of current nodes to original indices.

        Returns:

            FrozenSet[int]: A frozenset of original node indices.
        """
        s: Set[int] = set()
        for idx in np.where(community == comm_id)[0]:
            s.update(n2o[int(idx)])
            
        return frozenset(s)

    def _correlated_phase1(
        self,
        A: np.ndarray,
        k: np.ndarray,
        m: float,
        community: np.ndarray,
        n2o: Dict[int, Set[int]],
    ) -> Tuple[np.ndarray, int]:
        """
        Optimisation phase with combined modularity + correlation objective.

        Returns:

            Tuple[np.ndarray, int]: Updated community array and move count.
        """
        n = A.shape[0]
        total_moves = 0
        improved = True
        pass_num = 0

        while improved and pass_num < self.max_passes:
            improved = False
            pass_num += 1
            order = np.random.permutation(n)

            for node in order:
                cur_comm = int(community[node])
                nbr_idx = np.nonzero(A[node])[0]
                
                if len(nbr_idx) == 0:
                    continue

                candidate_comms = set(int(c) for c in community[nbr_idx])
                candidate_comms.add(cur_comm)

                stay_dQ = Louvain._delta_Q(node, cur_comm, community, A, k, m)

                cur_orig = self._collect_orig(cur_comm, community, n2o)
                node_orig = frozenset(n2o[node])
                corr_cur = self._pc1_correlation(cur_orig)
                corr_cur_without = self._pc1_correlation(cur_orig - node_orig)

                best_net = 0.0
                best_comm = cur_comm

                for tgt_comm in candidate_comms:
                    if tgt_comm == cur_comm:
                        continue

                    insert_dQ = Louvain._delta_Q(node, tgt_comm, community, A, k, m)
                    net_dQ = insert_dQ - stay_dQ

                    tgt_orig = self._collect_orig(tgt_comm, community, n2o)
                    corr_tgt = self._pc1_correlation(tgt_orig)
                    corr_tgt_with = self._pc1_correlation(tgt_orig | node_orig)

                    net_d_rho = (corr_cur_without + corr_tgt_with) - (corr_cur + corr_tgt)

                    net_combined = (self.k_L * net_dQ) + ((1.0 - self.k_L) * net_d_rho)

                    if net_combined > best_net + self.min_delta:
                        best_net = net_combined
                        best_comm = tgt_comm

                if best_comm != cur_comm:
                    best_orig = self._collect_orig(best_comm, community, n2o)
                    for key in (cur_orig, cur_orig - node_orig, best_orig, best_orig | node_orig):
                        self._corr_cache.pop(key, None)

                    community[node] = best_comm
                    improved = True
                    total_moves += 1

        logger.debug(f" Correlated Phase 1: {pass_num} passes, {total_moves} moves")
        return community, total_moves

    def run(self) -> Dict[Any, int]:
        """
        Execute the Correlated Louvain algorithm.

        Returns:

            Dict[Any, int]: Mapping of original nodes to community IDs.
        """
        self._corr_cache.clear()
        self._history.clear()

        A = self.A_orig.copy()
        degrees = A.sum(axis=1)
        m = A.sum() / 2.0
        community = np.arange(self.n, dtype=np.int64)
        n2o: Dict[int, Set[int]] = {i: {i} for i in range(self.n)}

        level = 0
        while True:
            level += 1
            logger.info(f"Level {level}: {A.shape[0]} nodes")

            community, moves = self._correlated_phase1(A, degrees, m, community, n2o)
            n_comms = len(np.unique(community))

            self._history.append({
                "level": level,
                "nodes_at_level": A.shape[0],
                "moves": moves,
                "communities": n_comms,
            })

            if moves == 0 or n_comms >= A.shape[0]:
                break

            A, degrees, m, community, n2o = Louvain._phase2(A, community, n2o)
            self._corr_cache.clear()

            if A.shape[0] <= 1:
                break

        partition: Dict[Any, int] = {}
        for si in range(len(community)):
            cid = int(community[si])
            for oi in n2o[si]:
                partition[self.idx_to_node[oi]] = cid

        remap = {old: new for new, old in enumerate(sorted(set(partition.values())))}
        self._partition = {nd: remap[c] for nd, c in partition.items()}
        self._modularity = self._compute_modularity()
        self._combined_quality = self._compute_combined_quality()

        logger.info(
            f"Correlated Louvain done: {len(remap)} communities, "
            f"Q={self._modularity:.4f}, Q*={self._combined_quality:.4f}"
        )
        return self._partition

    def _compute_combined_quality(self) -> float:
        """
        Compute the weighted combined quality score Q*.

        Returns:

            float: The score k_L * Q + (1 - k_L) * mean(|rho|).
        """
        Q = self._modularity if self._modularity is not None else 0.0
        corrs = []

        for cid, nds in self.communities.items():
            if len(nds) < 2:
                continue
            idx_set = frozenset(self.node_to_idx[n] for n in nds)
            corrs.append(self._pc1_correlation(idx_set))

        avg_rho = float(np.mean(corrs)) if corrs else 0.0
        return self.k_L * Q + (1.0 - self.k_L) * avg_rho

    def get_combined_quality(self) -> float:
        """
        Access the calculated combined quality score.

        Returns:

            float: The Q* score.
        """
        if self._combined_quality is None:
            raise ValueError("Call run() first.")
        return self._combined_quality

    def get_top_communities(self, n: int = 1) -> List[Tuple[int, float, List[Any]]]:
        """
        Retrieve the top communities based on absolute correlation.

        Args:

            n (int): Number of top communities to return.

        Returns:

            List[Tuple[int, float, List[Any]]]: Community data sorted by |rho|.
        """
        ranked = []
        for cid, nds in self.communities.items():
            if len(nds) < 2:
                continue
            idx_set = frozenset(self.node_to_idx[nd] for nd in nds)
            ranked.append((cid, self._pc1_correlation(idx_set), nds))
            
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:n]