import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from bioneuralnet.utils.logger import get_logger

# Try to import igraph + leidenalg
try:
    import igraph as ig
    import leidenalg
except ImportError as e:
    raise ImportError(
        "Leiden clustering requires `igraph` and `leidenalg`.\n"
        "Install them with: pip install igraph leidenalg"
    ) from e
class Leiden_upd:
    """
    Leiden community detection with an optional KMeans refinement step
    on top of node embeddings.

    If you just want vanilla Leiden, pass G and leave embeddings=None.
    If you pass embeddings, you can ask for refine_with_kmeans=True
    in .run() to split big communities into smaller ones in embedding space.
    """
    def __init__(
        self,
        G,
        embeddings=None,
        use_edge_weights: bool = True,
        resolution: float = 1.0,
        random_state: int | None = 0,
    ):
        self.logger = get_logger(__name__)
        self.G = G
        # keep a fixed node order so all label arrays match this
        self.nodes_ = list(G.nodes())

        self.use_edge_weights = use_edge_weights
        self.resolution = resolution
        self.random_state = random_state

        # embeddings are optional, only needed for the KMeans refinement
        self.embeddings = embeddings
        if embeddings is not None:
            if not isinstance(embeddings, np.ndarray):
                raise TypeError("embeddings should be a NumPy array (n_nodes, dim)")
            if embeddings.shape[0] != len(self.nodes_):
                # if this is wrong, everything else is kind of meaningless
                raise ValueError(
                    f"Embeddings have {embeddings.shape[0]} rows but graph has "
                    f"{len(self.nodes_)} nodes."
                )

        # build igraph version of the nx graph once; Leiden works on igraph
        self._ig_graph, self._weights = self._build_igraph_from_nx(
            self.G,
            use_edge_weights=self.use_edge_weights,
        )

        self._partition = None
        self.labels_leiden_ = None

        # keep old behavior: run Leiden right away so .labels_leiden_ is ready
        self.fit_leiden()

    # helper: NetworkX -> igraph. leiden allows weights so we use this to support building an igraph with weights
    def _build_igraph_from_nx(self, G, use_edge_weights: bool):
        """
        Convert a NetworkX graph to igraph.Graph and optionally set edge weights.
        """
        # just warning here
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            self.logger.warning(
                "Input is a MultiGraph/MultiDiGraph; parallel edges will be "
                "merged in igraph"
            )

        n = len(self.nodes_)
        node_index = {node: i for i, node in enumerate(self.nodes_)}

        directed = G.is_directed()
        ig_graph = ig.Graph(n=n, directed=directed)

        edges = []
        weights = []

        # map (u, v) in nx to integer ids in igraph, collect weights if needed
        for u, v, data in G.edges(data=True):
            iu = node_index[u]
            iv = node_index[v]
            edges.append((iu, iv))

            if use_edge_weights:
                raw = data.get("weight", 1.0)
                w = float(raw)
                if not np.isfinite(w):
                    raise ValueError(
                        f"Non-finite weight {raw!r} on edge ({u!r}, {v!r})"
                    )
                # negative weights are allowed mathematically but easy to mess up
                if w < 0:
                    self.logger.warning(
                        f"Negative weight {w} on edge ({u!r}, {v!r}). "
                        "Double-check if this is actually intended."
                    )
                weights.append(w)

        if edges:
            ig_graph.add_edges(edges)

        # if no weights or we chose not to use them treat as unweighted
        if not use_edge_weights or not edges:
            weights_attr = None
        else:
            if len(weights) != ig_graph.ecount():
                # this should never happen, so better hard fail
                raise RuntimeError(
                    f"Collected {len(weights)} weights for "
                    f"{ig_graph.ecount()} edges."
                )

            w_arr = np.asarray(weights, dtype=float)
            # if all weights are identical, it behaves like unweighted anyway
            if np.allclose(w_arr, w_arr[0]):
                self.logger.warning(
                    "use_edge_weights=True but all edge weights are the same "
                    f"({w_arr[0]}). This is effectively an unweighted run."
                )

            ig_graph.es["weight"] = weights
            weights_attr = "weight"

        self.logger.info(
            f"Built igraph from NetworkX: {ig_graph.vcount()} nodes, "
            f"{ig_graph.ecount()} edges, directed={directed}, "
            f"use_edge_weights={use_edge_weights}, "
            f"weights_effective={'yes' if weights_attr is not None else 'no'}."
        )

        return ig_graph, weights_attr

    # core Leiden call
    def fit_leiden(self):
        """
        Run the Leiden algorithm once and cache the base labels.
        """
        # RBConfigurationVertexPartition gives us a modularity-like objective
        partition = leidenalg.find_partition(
            self._ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=self._weights,
            resolution_parameter=self.resolution,
            seed=self.random_state,
        )

        self._partition = partition
        self.labels_leiden_ = np.asarray(partition.membership, dtype=int)

        n_comms = int(self.labels_leiden_.max()) + 1
        self.logger.info(
            f"Leiden found {n_comms} communities on "
            f"{len(self.nodes_)} nodes (resolution={self.resolution})."
        )
        return self

    # Return the partition quality (RB/modularity-like). None if not run.
    @property
    def modularity_(self):
        if self._partition is None:
            return None
        return float(self._partition.quality())

    def run(
        self,
        refine_with_kmeans: bool = True,
        max_k_per_community: int = 3,
    ) -> np.ndarray:
        """
        Get cluster labels.
        If refine_with_kmeans is False or no embeddings were given, this just returns the plain Leiden labels. Otherwise it will
        refine large communities using local KMeans in embedding space.
        """
        if self.labels_leiden_ is None:
            self.fit_leiden()

        # if no refinement requested (or no embeddings at all), bail early
        if not refine_with_kmeans or self.embeddings is None:
            return self.labels_leiden_.copy()

        labels_leiden = self.labels_leiden_
        labels_hybrid = np.full_like(labels_leiden, fill_value=-1, dtype=int)
        next_label = 0

        for com in np.unique(labels_leiden):
            idx = np.where(labels_leiden == com)[0]
            size = len(idx)

            # only try to split reasonably large communities
            # tiny ones tend to give junk splits and hurt modularity
            # size <= 2
            MIN_REFINE_SIZE = 30
            if size <= MIN_REFINE_SIZE:
                labels_hybrid[idx] = next_label
                next_label += 1
                continue

            # very simple rule: 1 subcluster per 10 points,
            # but never more than max_k_per_community.
            k_local = min(max_k_per_community, max(1, size // 10))
            sub_emb = self.embeddings[idx]

            if k_local == 1:
                # heuristic collapsed, just keep this community as-is
                labels_hybrid[idx] = next_label
                next_label += 1
                continue

            km_local = KMeans(
                n_clusters=k_local,
                random_state=self.random_state,
                n_init=10,
            ).fit(sub_emb)

            # assign a fresh global label for each local KMeans cluster
            for j in range(k_local):
                mask = (km_local.labels_ == j)
                if not np.any(mask):
                    # shouldnt really happen, but being safe
                    continue
                labels_hybrid[idx[mask]] = next_label
                next_label += 1

        # all nodes should have been assigned by now
        if (labels_hybrid == -1).any():
            raise RuntimeError(
                "Hybrid clustering left some nodes unlabeled, something went wrong.")
        n_clusters = int(labels_hybrid.max()) + 1
        self.logger.info(
            f"Hybrid Leiden+KMeans produced {n_clusters} clusters "
            f"(base communities: {self.labels_leiden_.max() + 1})."
        )

        return labels_hybrid
