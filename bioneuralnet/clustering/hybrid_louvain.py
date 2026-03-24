r"""
Hybrid Louvain-PageRank - Significant Subgraph Detection.

This module implements an iterative refinement algorithm that alternates 
between community detection (Louvain) and local ranking (PageRank) to 
identify phenotype-correlated subgraphs.

References:
    Abdel-Hafiz et al. (2022), "Significant Subgraph Detection in 
    Multi-omics Networks for Disease Pathway Identification," 
    Frontiers in Big Data.

Algorithm:
    The process alternates between global community detection and local 
    refinement until convergence:

    Iteration 1 (Global Scope):
        1. Run Correlated Louvain on the full graph to optimize Hybrid Modularity.
        2. Select the community with the highest phenotype correlation :math:`|\rho|`.
        3. Assign seed weights to these nodes based on their marginal 
           contribution to :math:`\rho`.
        4. Execute Correlated PageRank on the full graph.
        5. Use a sweep cut to produce the initial refined subgraph.

    Iteration 2+ (Local Scope):
        1. Restrict the graph strictly to the output of the previous PageRank.
        2. Run Correlated Louvain on this reduced subgraph.
        3. Repeat refinement steps until size converges or a singleton is produced.

    Output:
        The subgraph that achieved the highest :math:`|\rho|` across all iterations.

Notes:
    **Hybrid Modularity (Correlated Louvain)**
    Balances internal topological connectivity with phenotype correlation:
    
    .. math::
        Q_{hybrid} = k_L Q + (1 - k_L) \rho
    
    **Hybrid Conductance (Correlated PageRank)**
    Balances the external cut/internal volume ratio with correlation:
    
    .. math::
        \Phi_{hybrid} = k_P \Phi + (1 - k_P) \rho

    **Seed Weighting**
    Teleportation probabilities :math:`\alpha_i` are weighted by a node's 
    marginal contribution:
    
    .. math::
        \alpha_i = \frac{\rho_i}{\max(\rho_{seeds})} \cdot \alpha_{max}
    
    Where :math:`\rho_i = \rho(S) - \rho(S \setminus \{i\})`.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import pandas as pd

from .correlated_louvain import CorrelatedLouvain
from .correlated_pagerank import CorrelatedPageRank
from ..utils import get_logger

logger = get_logger(__name__)

class HybridLouvain:
    """
    Hybrid Louvain-PageRank for significant subgraph detection.

    Iteratively refines a multi-omics network by alternating:

    (a) Correlated Louvain to find the most phenotype-associated community
    (b) Correlated PageRank to refine that community via sweep cut

    The graph shrinks each iteration. The best subgraph by |rho| is
    tracked across all iterations and returned.

    Args:

        G (Union[nx.Graph, pd.DataFrame]): Weighted undirected graph or adjacency matrix DataFrame.
        B (pd.DataFrame): Omics data (n_samples x n_features).
        Y (Union[pd.DataFrame, pd.Series]): Phenotype vector.
        k_L (float): Weight on modularity for Correlated Louvain).
        teleport_prob (float): Teleportation probability for PageRank (alpha).
        k_P (float): Weight on conductance for PageRank sweep cut.
        max_iter (int): Maximum Hybrid iterations.
        min_nodes (int): Stop if graph shrinks below this size.
        weight (str): Edge attribute name for weights.
        seed (Optional[int]): Random seed.

    """

    def __init__(
        self,
        G: Union[nx.Graph, pd.DataFrame],
        B: pd.DataFrame,
        Y: Union[pd.DataFrame, pd.Series],
        k_L: float = 0.8,
        teleport_prob: float = 0.05,
        k_P: float = 0.7,
        max_iter: int = 10,
        min_nodes: int = 3,
        weight: str = "weight",
        seed: Optional[int] = None,
    ):
        if isinstance(G, pd.DataFrame):
            logger.info("Converting adjacency DataFrame to NetworkX graph.")
            G = nx.from_pandas_adjacency(G)
        
        if not isinstance(G, nx.Graph):
            raise TypeError("G must be a networkx.Graph or adjacency DataFrame.")

        self.G_original = G.copy()
        self.weight = weight

        graph_nodes = set(str(n) for n in G.nodes())
        keep = [c for c in B.columns if str(c) in graph_nodes]
        dropped = len(B.columns) - len(keep)
        
        if dropped > 0:
            logger.info(f"Dropped {dropped} omics columns not in graph.")
        
        self.B = B.loc[:, keep].copy()

        if isinstance(Y, pd.DataFrame):
            self.Y = Y.squeeze()
        elif isinstance(Y, pd.Series):
            self.Y = Y
        else:
            self.Y = pd.Series(Y)

        self.k_L = k_L
        self.teleport_prob = teleport_prob
        self.k_P = k_P
        self.max_iter = max_iter
        self.min_nodes = min_nodes
        self.seed = seed

        self._iterations: List[Dict[str, Any]] = []
        self._best_idx: Optional[int] = None

        logger.info(
            f"HybridLouvain: nodes={G.number_of_nodes()}, "
            f"edges={G.number_of_edges()}, features={self.B.shape[1]}, "
            f"k_L={k_L}, teleport={teleport_prob}, k_P={k_P}, "
            f"max_iter={max_iter}"
        )

    def run(self, as_dfs: bool = False) -> Union[Dict[str, Any], List[pd.DataFrame]]:
        """
        Execute the Hybrid Louvain-PageRank algorithm.

        Returns:

            Dict:

                - best_nodes: nodes of the highest |rho| subgraph
                - best_correlation: float
                - best_iteration: int
                - iterations: full per-iteration metadata
                - all_subgraphs: {iteration_index: [nodes]}

        """
        self._iterations.clear()
        self._best_idx = None

        G_remaining = self.G_original.copy()
        prev_size = G_remaining.number_of_nodes()

        for it in range(self.max_iter):
            n_nodes = G_remaining.number_of_nodes()
            n_edges = G_remaining.number_of_edges()

            logger.info(f"\n--- Iteration {it + 1}: {n_nodes} nodes remaining ---")

            if n_nodes < self.min_nodes:
                logger.info(f"Remaining graph has {n_nodes} < {self.min_nodes} nodes. Done.")
                break

            # part 1: running Correlated Louvain on remaining graph
            try:
                louvain = CorrelatedLouvain(
                    G=G_remaining,
                    B=self.B,
                    Y=self.Y,
                    k_L=self.k_L,
                    weight=self.weight,
                    seed=self.seed,
                )
                partition_local = louvain.run()
            except Exception as e:
                logger.error(f"Louvain failed at iteration {it + 1}: {e}")
                break

            top = louvain.get_top_communities(n=1)

            if not top:
                logger.info("No non-singleton communities found. Done.")
                break

            top_cid, top_rho, seed_nodes = top[0]

            if len(seed_nodes) < 2:
                logger.info("Top community is a singleton. Done.")
                break

            logger.info(
                f"Top community: |rho|={top_rho:.4f}, size={len(seed_nodes)}"
            )

            # part 2: Correlated PageRank refines
            try:
                pagerank = CorrelatedPageRank(
                    graph=G_remaining,
                    omics_data=self.B,
                    phenotype_data=self.Y,
                    teleport_prob=self.teleport_prob,
                    k_P=self.k_P,
                    min_cluster=self.min_nodes,
                    seed=self.seed,
                )
                pr_results = pagerank.run(seed_nodes)
            except Exception as e:
                logger.error(f"PageRank failed at iteration {it + 1}: {e}")
                break

            refined_nodes = pr_results.get("cluster_nodes", [])
            new_size = len(refined_nodes)

            if len(refined_nodes) < 2:
                logger.info("PageRank produced < 2 nodes. Done.")
                break

            refined_rho = abs(pagerank.phen_omics_corr(refined_nodes)[0])
            if self._best_idx is None or refined_rho > self._iterations[self._best_idx]["refined_rho"]:
                self._best_idx = it

            # save this communit
            iter_result = {
                "iteration": it,
                "graph_nodes": n_nodes,
                "graph_edges": n_edges,
                "louvain_communities": len(set(partition_local.values())),
                "louvain_quality": louvain.get_combined_quality(),
                "seed_size": len(seed_nodes),
                "seed_rho": top_rho,
                "refined_size": len(refined_nodes),
                "refined_rho": refined_rho,
                "conductance": pr_results.get("conductance"),
                "composite_score": pr_results.get("composite_score"),
                "refined_nodes": list(refined_nodes),
            }
            self._iterations.append(iter_result)

            logger.info(
                f"  Iteration {it}: {len(refined_nodes)} nodes, "
                f"|rho|={refined_rho:.4f}"
            )

            # zoom-in: next iteration operates on the refined subgraph
            if new_size == prev_size:
                logger.info("Subgraph size unchanged; stopping.")
                break
            prev_size = new_size
            G_remaining = G_remaining.subgraph(refined_nodes).copy()

        best_nodes = self._iterations[self._best_idx]["refined_nodes"] if self._best_idx is not None else []
        best_rho = self._iterations[self._best_idx]["refined_rho"] if self._best_idx is not None else 0.0

        logger.info(f"\nHybrid Louvain completed: {len(self._iterations)} iterations, best |rho|={best_rho:.4f} at iteration {self._best_idx}")
        logger.info("All iterations:")
        for it_r in self._iterations:
            logger.info(
                f"  iter {it_r['iteration']}: size={it_r['refined_size']}, "
                f"|rho|={it_r['refined_rho']:.4f}, conductance={it_r['conductance']:.4f}"
            )
        logger.info(f"  Best subgraph: {len(best_nodes)} nodes, |rho|={best_rho:.4f} (iter {self._best_idx})")

        if as_dfs:
            result = []
            for it_r in self._iterations:
                nodes = it_r["refined_nodes"]
                sub_G = self.G_original.subgraph(nodes)
                adj = nx.to_pandas_adjacency(sub_G, weight=self.weight)
                result.append(adj)
            return result

        return {
            "best_nodes": best_nodes,
            "best_correlation": best_rho,
            "best_iteration": self._best_idx,
            "iterations": self._iterations,
            "all_subgraphs": {it_r["iteration"]: it_r["refined_nodes"] for it_r in self._iterations},
        }

    @property
    def iterations(self) -> List[Dict[str, Any]]:
        """
        Provides access to per-iteration details from the most recent run.

        Returns:

            List[Dict[str, Any]]: A list of result dictionaries for each iteration.
        """
        return list(self._iterations)

    @property
    def best_subgraph(self) -> Tuple[List[Any], float, int]:
        """
        Retrieves the nodes and performance metrics of the best subgraph found.

        Returns:

            Tuple[List[Any], float, int]: (nodes, |rho|, iteration_index).
        """
        if self._best_idx is None:
            raise ValueError("Call run() first.")
        
        best = self._iterations[self._best_idx]
        
        return best["refined_nodes"], best["refined_rho"], best["iteration"]