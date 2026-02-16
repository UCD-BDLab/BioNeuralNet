"""
Correlated PageRank Clustering.

This module implements a personalized PageRank algorithm combined with a 
phenotype-aware sweep cut to detect significant subgraphs.

References:
    Abdel-Hafiz et al. (2022), "Significant Subgraph Detection in 
    Multi-omics Networks for Disease Pathway Identification," 
    Frontiers in Big Data.

Algorithm:
    The PageRank vector is computed as the stationary distribution of:
    
    .. math::
        pr_{\\alpha}(s) = \\alpha s + (1 - \\alpha) pr_{\\alpha}(s) W

    Where:
        * :math:`\\alpha`: Teleportation (restart) probability.
        * :math:`s`: Personalization vector (seed weights).
        * :math:`W`: Transition matrix.

    .. important::
        The `networkx.pagerank` implementation uses a `alpha` parameter 
        representing the **damping factor** (link-following probability). 
        Therefore, :math:`\\text{nx_alpha} = 1 - \\alpha_{theoretical}`.

Notes:
    **Sweep Cut Optimization**
    Nodes are sorted by PageRank-per-degree in descending order. For each 
    prefix set :math:`S_i`, the algorithm minimizes the **Hybrid Conductance**:

    .. math::
        \\Phi_{hybrid} = k_P \\Phi + (1 - k_P) \\rho

    Where:
        * :math:`\\Phi`: Standard conductance (:math:`cut / \min(vol(S), vol(V \setminus S))`).
        * :math:`\\rho`: Negative absolute Pearson correlation (:math:`-|\\rho|`).
        * :math:`k_P`: Trade-off weight (Default: ~0.5).

    **Personalization Vector (Seed Weighting)**
    Teleportation probabilities for seeds are weighted by their marginal 
    contribution to correlation:

    .. math::
        \\alpha_i = \\frac{\\rho_i}{\\max(\\rho_{seeds})} \\cdot \\alpha_{max}

    Where :math:`\\rho_i = |\\rho(S)| - |\\rho(S \setminus \{i\})|`. 
    Values where :math:`\\rho_i < 0` are clamped to 0.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CorrelatedPageRank:
    """
    Correlated PageRank clustering on a multi-omics network.

    Args:

        graph (nx.Graph): Weighted undirected NetworkX graph.
        omics_data (pd.DataFrame): Omics matrix (n_samples x n_features), columns = node ids.
        phenotype_data (Union[pd.DataFrame, pd.Series]): Phenotype vector aligned with rows of omics_data.
        teleport_prob (float): Teleportation probability (alpha). Default 0.10.
        k_P (float): Weight on conductance in combined objective (Eq. 5).
        max_iter (int): Max iterations for PageRank power iteration.
        tol (float): Convergence tolerance for PageRank.
        min_cluster (int): Minimum cluster size for sweep cut consideration.
        seed (Optional[int]): Random seed for reproducibility.
    """

    def __init__(
        self,
        graph: nx.Graph,
        omics_data: pd.DataFrame,
        phenotype_data: Union[pd.DataFrame, pd.Series],
        teleport_prob: float = 0.10,
        k_P: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        min_cluster: int = 2,
        seed: Optional[int] = None,
    ):
        self.G = graph
        self.B = omics_data

        if isinstance(phenotype_data, pd.DataFrame):
            self.Y = phenotype_data.squeeze()
        elif isinstance(phenotype_data, pd.Series):
            self.Y = phenotype_data
        else:
            self.Y = pd.Series(phenotype_data)

        if not 0.0 <= teleport_prob <= 1.0:
            raise ValueError(f"teleport_prob must be in [0, 1], got {teleport_prob}")
            
        self.teleport_prob = teleport_prob
        self._nx_alpha = 1.0 - teleport_prob

        if not 0.0 <= k_P <= 1.0:
            raise ValueError(f"k_P must be in [0, 1], got {k_P}")
            
        self.k_P = k_P
        self.max_iter = max_iter
        self.tol = tol
        self.min_cluster = min_cluster

        if seed is not None:
            np.random.seed(seed)

        self._validate_inputs()

        logger.info(
            f"CorrelatedPageRank: nodes={self.G.number_of_nodes()}, "
            f"teleport={teleport_prob} (nx_alpha={self._nx_alpha}), k_P={k_P}"
        )

    def _validate_inputs(self):
        """
        Check input consistency across graph and omics data.

        Raises:

            TypeError: If graph or omics_data are not of the expected types.
        """
        if not isinstance(self.G, nx.Graph):
            raise TypeError("graph must be a networkx.Graph")
            
        if not isinstance(self.B, pd.DataFrame):
            raise TypeError("omics_data must be a pandas DataFrame")
            
        graph_nodes = set(str(n) for n in self.G.nodes())
        omics_cols = set(str(c) for c in self.B.columns)
        missing = graph_nodes - omics_cols
        
        if missing:
            logger.warning(
                f"{len(missing)} graph nodes missing from omics columns "
                f"(first 5: {sorted(missing)[:5]})"
            )

    def phen_omics_corr(self, nodes: List[Any]) -> Tuple[float, float]:
        """
        Compute Pearson(PC1(omics[:, nodes]), phenotype).

        Args:

            nodes (List[Any]): List of node identifiers.

        Returns:

            Tuple[float, float]: (correlation, p_value). Returns (0.0, 1.0) on failure.
        """
        valid_cols = []
        for n in nodes:
            if n in self.B.columns:
                valid_cols.append(n)
            else:
                n_str = str(n)
                if n_str in self.B.columns:
                    valid_cols.append(n_str)

        if len(valid_cols) < 2:
            return 0.0, 1.0

        B_sub = self.B[valid_cols]
        
        if B_sub.shape[0] < 2:
            return 0.0, 1.0

        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(B_sub)

            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(scaled).flatten()

            if isinstance(self.Y, pd.Series):
                common_idx = B_sub.index.intersection(self.Y.index)
                if len(common_idx) < 2:
                    return 0.0, 1.0
                pc1 = pc1[:len(common_idx)]
                y_vals = self.Y.loc[common_idx].values
            else:
                y_vals = np.asarray(self.Y)
                n_limit = min(len(pc1), len(y_vals))
                if n_limit < 2:
                    return 0.0, 1.0
                pc1, y_vals = pc1[:n_limit], y_vals[:n_limit]

            corr, pvalue = pearsonr(pc1, y_vals)
            
            return (float(corr), float(pvalue)) if np.isfinite(corr) else (0.0, 1.0)

        except Exception as e:
            logger.error(f"phen_omics_corr error: {e}")
            return 0.0, 1.0

    def sweep_cut(self, pr_scores: Dict[Any, float]) -> Dict[str, Any]:
        """
        Identify the best cluster via sweep cut on PageRank scores.

        Args:

            pr_scores (Dict[Any, float]): Mapping of nodes to PageRank scores.

        Returns:

            Dict: Best cluster details including nodes, conductance, and composite score.
        """
        degrees = dict(self.G.degree(weight="weight"))

        vec = sorted(
            [
                (pr_scores[nd] / degrees[nd] if degrees[nd] > 0 else 0.0, nd)
                for nd in pr_scores
            ],
            reverse=True,
        )

        best_result = {
            "cluster_nodes": [],
            "cluster_size": 0,
            "conductance": 1.0,
            "correlation": 0.0,
            "correlation_pvalue": 1.0,
            "composite_score": float("inf"),
        }

        all_nodes = set(self.G.nodes())
        current_cluster = set()

        for val, node in vec:
            if val == 0:
                continue

            current_cluster.add(node)
            complement = all_nodes - current_cluster

            if len(current_cluster) >= len(all_nodes) or len(complement) == 0:
                continue

            if len(current_cluster) < self.min_cluster:
                continue

            vol_S = sum(d for _, d in self.G.degree(current_cluster, weight="weight"))
            vol_T = sum(d for _, d in self.G.degree(complement, weight="weight"))
            
            if min(vol_S, vol_T) == 0:
                continue

            cond = nx.conductance(self.G, current_cluster, weight="weight")
            corr, pval = self.phen_omics_corr(list(current_cluster))
            composite = self.k_P * cond + (1.0 - self.k_P) * (-abs(corr))

            if composite < best_result["composite_score"]:
                best_result = {
                    "cluster_nodes": list(current_cluster),
                    "cluster_size": len(current_cluster),
                    "conductance": round(cond, 4),
                    "correlation": round(corr, 4),
                    "correlation_pvalue": pval,
                    "composite_score": round(composite, 4),
                }

        if not best_result["cluster_nodes"]:
            logger.warning("Sweep cut found no valid cluster.")

        return best_result

    def generate_weighted_personalization(
        self,
        nodes: List[Any],
        alpha_max: Optional[float] = None,
    ) -> Dict[Any, float]:
        """
        Build personalization vector based on each node's correlation contribution.

        Args:

            nodes (List[Any]): Seed node list.
            alpha_max (Optional[float]): Maximum teleportation weight.

        Returns:

            Dict[Any, float]: Personalization mapping {node: weight}.
        """
        if not nodes:
            return {}

        if alpha_max is None:
            alpha_max = self.teleport_prob

        total_corr, _ = self.phen_omics_corr(nodes)
        abs_total = abs(total_corr)

        contributions = []
        for i in range(len(nodes)):
            nodes_excl = nodes[:i] + nodes[i + 1:]
            if not nodes_excl:
                contributions.append(0.0)
                continue
                
            corr_excl, _ = self.phen_omics_corr(nodes_excl)
            rho_i = abs_total - abs(corr_excl)
            contributions.append(rho_i)

        floor = 1e-4 * alpha_max
        clamped = [max(c, floor) for c in contributions]
        max_contrib = max(clamped)

        if max_contrib > 0:
            personalization = {
                nodes[i]: (clamped[i] / max_contrib) * alpha_max
                for i in range(len(nodes))
            }
        else:
            uniform = 1.0 / len(nodes)
            personalization = {nd: uniform for nd in nodes}

        return personalization

    def run(self, seed_nodes: List[Any]) -> Dict[str, Any]:
        """
        Execute Correlated PageRank clustering.

        Args:

            seed_nodes (List[Any]): Nodes to use as the teleport set.

        Returns:

            Dict: Cluster performance and node list.
        """
        if not seed_nodes:
            raise ValueError("seed_nodes cannot be empty.")

        graph_nodes = set(self.G.nodes())
        missing = set(seed_nodes) - graph_nodes
        
        if missing:
            raise ValueError(f"Seed nodes not in graph: {missing}")

        personalization = self.generate_weighted_personalization(seed_nodes)
        
        logger.info(
            f"Personalization: {len(personalization)} nodes, "
            f"max_weight={max(personalization.values()):.4f}, "
            f"min_weight={min(personalization.values()):.4f}"
        )

        try:
            pr_scores = nx.pagerank(
                self.G,
                alpha=self._nx_alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol,
                weight="weight",
            )
        except nx.exception.PowerIterationFailedConvergence:
            logger.warning(f"PageRank did not converge; retrying with {self.max_iter * 2} iters.")
            pr_scores = nx.pagerank(
                self.G,
                alpha=self._nx_alpha,
                personalization=personalization,
                max_iter=self.max_iter * 2,
                tol=self.tol,
                weight="weight",
            )

        logger.info("PageRank computation completed.")
        results = self.sweep_cut(pr_scores)

        if results["cluster_nodes"]:
            logger.info(
                f"Sweep cut: size={results['cluster_size']}, "
                f"cond={results['conductance']}, "
                f"corr={results['correlation']}, "
                f"composite={results['composite_score']}"
            )
        else:
            logger.warning("Sweep cut found no valid cluster.")

        return results