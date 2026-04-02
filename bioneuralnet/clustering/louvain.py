r"""
Standard Louvain Method for Community Detection - NumPy Implementation.

References:
    Blondel et al. (2008), "Fast unfolding of communities in large networks."
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

class Louvain:
    """
    Standard Louvain community detection.

    This class encapsulates the multi-phase optimization algorithm for detecting communities in weighted graphs.
    """

    def __init__(
        self,
        G: nx.Graph,
        weight: str = "weight",
        max_passes: int = 100,
        min_delta: float = 1e-10,
        seed: Optional[int] = None,
    ):
        """
        Initializes the Louvain algorithm with the graph and optimization parameters.

        Sets up the internal adjacency structures and seeds the random number generator if specified.

        Args:

            G (nx.Graph): The input weighted undirected NetworkX graph to be analyzed.
            weight (str): The edge attribute name representing weights to use for the adjacency matrix.
            max_passes (int): The maximum number of full-node sweeps allowed per local optimization phase.
            min_delta (float): The minimum modularity gain required to accept a node movement.
            seed (Optional[int]): A random seed used for reproducibility of the node traversal order.

        """

        self.weight_attr = weight
        self.max_passes = max_passes
        self.min_delta = min_delta

        if seed is not None:
            np.random.seed(seed)

        self.nodes: List[Any] = sorted(G.nodes())
        self.n: int = len(self.nodes)
        self.node_to_idx = {nd: i for i, nd in enumerate(self.nodes)}
        self.idx_to_node = {i: nd for i, nd in enumerate(self.nodes)}

        A = np.zeros((self.n, self.n), dtype=np.float64)
        for u, v, data in G.edges(data=True):
            w = float(data.get(weight, 1.0))
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            if i == j:
                A[i, i] = w
            else:
                A[i, j] = w
                A[j, i] = w
        self.A_orig = A

        self._partition: Optional[Dict[Any, int]] = None
        self._modularity: Optional[float] = None
        self._history: List[Dict[str, Any]] = []

        logger.info(f"Louvain: {self.n} nodes, "
                     f"{G.number_of_edges()} edges")

    @staticmethod
    def _delta_Q(
        node: int,
        target_comm: int,
        community: np.ndarray,
        A: np.ndarray,
        k: np.ndarray,
        m: float,
    ) -> float:
        """
        Calculates the modularity gain (DeltaQ) for placing an isolated node into a target community.

        This uses the formula DeltaQ = k_{v,in} / m - Sigma_tot * k_v / (2 * m^2) where terms are computed excluding the node itself.

        Args:

            node (int): The integer index of the node being considered for movement.
            target_comm (int): The integer ID of the candidate community.
            community (np.ndarray): An array mapping every node index to its current community ID.
            A (np.ndarray): The adjacency matrix representing the graph structure and weights.
            k (np.ndarray): An array containing the weighted degree (strength) of each node.
            m (float): The total weight of all edges in the graph.

        Returns:

            float: The calculated change in modularity if the node were added to the target community.
        """
        mask = (community == target_comm)
        mask = mask.copy()
        mask[node] = False

        k_v_in = A[node][mask].sum()

        sigma_tot = k[mask].sum()

        k_v = k[node]

        return k_v_in / m - sigma_tot * k_v / (2.0 * m * m)

    @staticmethod
    def _phase1(
        A: np.ndarray,
        k: np.ndarray,
        m: float,
        community: np.ndarray,
        max_passes: int,
        min_delta: float,
    ) -> Tuple[np.ndarray, int]:
        """
        Executes the local optimization phase by moving nodes between communities to maximize modularity.

        Iteratively evaluates moving each node to neighbor communities and applies the move that yields the highest positive net gain.

        Args:

            A (np.ndarray): The current adjacency matrix of the graph or super-graph.
            k (np.ndarray): An array of node degrees corresponding to the current adjacency matrix.
            m (float): The total edge weight of the network.
            community (np.ndarray): The current community assignment for each node.
            max_passes (int): The limit on the number of iterations over all nodes.
            min_delta (float): The threshold for modularity improvement required to trigger a move.

        Returns:

            Tuple[np.ndarray, int]: A tuple containing the updated community assignments and the total number of moves performed.
        """
        n = A.shape[0]
        total_moves = 0
        improved = True
        pass_num = 0

        while improved and pass_num < max_passes:
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

                best_dQ = 0.0
                best_comm = cur_comm

                stay_dQ = Louvain._delta_Q(
                    node, cur_comm, community, A, k, m
                )

                for c in candidate_comms:
                    if c == cur_comm:
                        continue

                    insert_dQ = Louvain._delta_Q(
                        node, c, community, A, k, m
                    )

                    net = insert_dQ - stay_dQ

                    if net > best_dQ + min_delta:
                        best_dQ = net
                        best_comm = c

                if best_comm != cur_comm:
                    community[node] = best_comm
                    improved = True
                    total_moves += 1

        logger.debug(f"  Phase 1: {pass_num} passes, {total_moves} moves")
        return community, total_moves

    @staticmethod
    def _phase2(
        A: np.ndarray,
        community: np.ndarray,
        n2o: Dict[int, Set[int]],
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, Dict[int, Set[int]]]:
        """
        Aggregates the graph based on the community structure found in Phase 1 to create super-nodes.

        Compresses the adjacency matrix such that new nodes represent previous communities and edge weights are summed.

        Args:

            A (np.ndarray): The adjacency matrix of the graph before aggregation.
            community (np.ndarray): The community assignments for the nodes in the current graph.
            n2o (Dict[int, Set[int]]): A mapping from current node indices to the set of original node indices they represent.

        Returns:

            Tuple[np.ndarray, np.ndarray, float, np.ndarray, Dict[int, Set[int]]]: A tuple containing the new adjacency matrix, degrees, total weight, initial community vector, and updated node mapping.
        """
        unique_comms = np.unique(community)
        n_new = len(unique_comms)
        comm_map = {int(c): i for i, c in enumerate(unique_comms)}
        n_old = A.shape[0]

        mapping = np.array([comm_map[int(community[i])] for i in range(n_old)])
        M = np.zeros((n_old, n_new), dtype=np.float64)
        M[np.arange(n_old), mapping] = 1.0

        new_A = M.T @ A @ M

        new_k = new_A.sum(axis=1)
        new_m = new_A.sum() / 2.0

        new_n2o: Dict[int, Set[int]] = {}
        for old_idx in range(n_old):
            new_idx = mapping[old_idx]
            new_n2o.setdefault(new_idx, set()).update(n2o[old_idx])

        new_community = np.arange(n_new, dtype=np.int64)

        logger.debug(f"  Phase 2: {n_old} -> {n_new} super-nodes")
        return new_A, new_k, new_m, new_community, new_n2o

    def run(self) -> Dict[Any, int]:
        """
        Executes the full Louvain algorithm by alternating between local optimization and graph aggregation.

        Loops until the modularity converges or the graph cannot be aggregated further.

        Returns:

            Dict[Any, int]: A dictionary mapping original node identifiers to their final community IDs.
        """
        self._history.clear()

        A = self.A_orig.copy()
        k = A.sum(axis=1)
        m = A.sum() / 2.0
        community = np.arange(self.n, dtype=np.int64)
        n2o: Dict[int, Set[int]] = {i: {i} for i in range(self.n)}

        level = 0
        while True:
            level += 1
            logger.info(f"Level {level}: {A.shape[0]} nodes")

            community, moves = self._phase1(
                A, k, m, community, self.max_passes, self.min_delta
            )
            n_comms = len(np.unique(community))

            self._history.append({
                "level": level,
                "nodes_at_level": A.shape[0],
                "moves": moves,
                "communities": n_comms,
            })

            if moves == 0 or n_comms >= A.shape[0]:
                break

            A, k, m, community, n2o = self._phase2(A, community, n2o)

            if A.shape[0] <= 1:
                break

        partition: Dict[Any, int] = {}
        for super_idx in range(len(community)):
            cid = int(community[super_idx])
            for orig_idx in n2o[super_idx]:
                partition[self.idx_to_node[orig_idx]] = cid

        remap = {old: new for new, old in enumerate(sorted(set(partition.values())))}
        self._partition = {nd: remap[c] for nd, c in partition.items()}

        self._modularity = self._compute_modularity()

        logger.info(
            f"Louvain done: {len(remap)} communities, "
            f"Q = {self._modularity:.6f}, levels = {level}"
        )
        return self._partition

    def _compute_modularity(self) -> float:
        """
        Computes the final modularity score (Q) of the partition on the original graph.

        This serves as a verification step ensuring the result aligns with the standard modularity definition.

        Returns:

            float: The modularity score of the final partition.
        """
        if self._partition is None:
            return 0.0

        A = self.A_orig
        k = A.sum(axis=1)
        m = A.sum() / 2.0
        if m == 0:
            return 0.0

        comm = np.array([self._partition[self.idx_to_node[i]] for i in range(self.n)])

        Q = 0.0
        for i in range(self.n):
            for j in range(self.n):
                if comm[i] == comm[j]:
                    Q += A[i, j] - k[i] * k[j] / (2.0 * m)

        return Q / (2.0 * m)

    @property
    def partition(self) -> Dict[Any, int]:
        """
        Retrieves the final partition of the graph.

        Requires that the run() method has been executed previously.

        Returns:

            Dict[Any, int]: A dictionary mapping nodes to community IDs.
        """
        if self._partition is None:
            raise ValueError("Call run() first.")
        return self._partition

    @property
    def communities(self) -> Dict[int, List[Any]]:
        """
        Retrieves the communities grouped by community ID.

        Convenient for iterating over sets of nodes belonging to the same community.

        Returns:

            Dict[int, List[Any]]: A dictionary mapping community IDs to lists of nodes.
        """
        d: Dict[int, List[Any]] = {}
        for nd, cid in self.partition.items():
            d.setdefault(cid, []).append(nd)
        return d

    @property
    def modularity(self) -> float:
        """
        Retrieves the final modularity score of the computed partition.

        Requires that the run() method has been executed previously.

        Returns:

            float: The modularity score.
        """
        if self._modularity is None:
            raise ValueError("Call run() first.")
        return self._modularity

    @property
    def history(self) -> List[Dict[str, Any]]:
        """
        Retrieves the history of the algorithm's execution levels.

        Provides insight into the convergence process and reduction of graph size.

        Returns:

            List[Dict[str, Any]]: A list of dictionaries containing stats for each level.
        """
        return list(self._history)
