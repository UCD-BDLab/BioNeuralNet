import sys
import types
import networkx as nx
import numpy as np
import pytest


# Inject fake `igraph` and `leidenalg` modules so the module-level import
# in `leiden_upd.py` succeeds and we can control partition outputs.
def _install_fake_leiden_modules():
    import types

    fake_igraph = types.ModuleType("igraph")

    class FakeGraph:
        def __init__(self, n=0, directed=False):
            self._n = int(n)
            self._directed = bool(directed)
            self._edges = []
            # mimic igraph edge-sequence attribute storage
            self.es = {}

        def add_edges(self, edges):
            # edges is an iterable of (u,v) tuples
            self._edges.extend(edges)

        def vcount(self):
            return self._n

        def ecount(self):
            return len(self._edges)

    fake_igraph.Graph = FakeGraph

    fake_leidenalg = types.ModuleType("leidenalg")

    class FakePartition:
        def __init__(self, n_nodes, membership=None):
            if membership is None:
                membership = [0] * int(n_nodes)
            self.membership = membership

        def quality(self):
            return 0.42

    def find_partition(graph, partition_type, weights=None, resolution_parameter=None, seed=None):
        # Return a partition where all nodes belong to community 0
        return FakePartition(graph.vcount())

    fake_leidenalg.find_partition = find_partition
    fake_leidenalg.RBConfigurationVertexPartition = object

    sys.modules.setdefault("igraph", fake_igraph)
    sys.modules.setdefault("leidenalg", fake_leidenalg)


_install_fake_leiden_modules()

from bioneuralnet.clustering.leiden_upd import Leiden_upd


def test_basic_leiden_builds_igraph_and_runs_partition():
    G = nx.path_graph(5)
    # path_graph edges have no 'weight' attribute, default 1.0 will be used
    inst = Leiden_upd(G=G)

    assert inst._ig_graph.vcount() == 5
    assert inst._ig_graph.ecount() == 4
    # weights were collected (defaults) so attribute name should be set
    assert inst._weights == "weight"
    # labels should be present and length match nodes
    assert inst.labels_leiden_.shape[0] == 5
    # fake partition returns all zeros
    assert int(inst.labels_leiden_.max()) == 0
    # modularity_ should proxy to fake partition quality
    assert pytest.approx(inst.modularity_, rel=1e-6) == 0.42


def test_run_returns_copy_when_no_refine():
    G = nx.path_graph(3)
    inst = Leiden_upd(G=G)
    out = inst.run(refine_with_kmeans=False)
    # same values, but ensure a copy (not the same object)
    assert np.array_equal(out, inst.labels_leiden_)
    assert out is not inst.labels_leiden_


def test_graph_with_no_edges_sets_weights_none():
    G = nx.Graph()
    G.add_nodes_from(range(3))
    inst = Leiden_upd(G=G)
    assert inst._ig_graph.ecount() == 0
    assert inst._weights is None
    assert inst.labels_leiden_.shape[0] == 3


def test_non_finite_weight_raises_value_error():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=float("nan"))
    with pytest.raises(ValueError):
        Leiden_upd(G=G)


def test_negative_weight_logs_warning_but_succeeds():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=-2.5)
    inst = Leiden_upd(G=G)
    # negative weight should not raise; weights attribute is set
    assert inst._weights == "weight"


def test_embeddings_type_and_size_validation():
    G = nx.path_graph(4)
    # wrong type
    with pytest.raises(TypeError):
        Leiden_upd(G=G, embeddings=[1, 2, 3, 4])

    # wrong size
    bad_emb = np.zeros((2, 3))
    with pytest.raises(ValueError):
        Leiden_upd(G=G, embeddings=bad_emb)


def test_refine_with_kmeans_splits_large_community():
    # create a graph with many nodes; fake partition will put all nodes into one community
    n = 60
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # add a few edges to make ecount > 0
    for i in range(n - 1):
        G.add_edge(i, i + 1)

    # create embeddings with 3 clear clusters so KMeans can split
    emb = np.vstack([
        np.random.randn(n // 3, 2) + np.array([0.0, 0.0]),
        np.random.randn(n // 3, 2) + np.array([10.0, 0.0]),
        np.random.randn(n - 2 * (n // 3), 2) + np.array([0.0, 10.0]),
    ])

    inst = Leiden_upd(G=G, embeddings=emb, random_state=0)
    labels_hybrid = inst.run(refine_with_kmeans=True, max_k_per_community=3)
    assert labels_hybrid.shape[0] == n
    # should produce more than one cluster
    assert len(np.unique(labels_hybrid)) >= 2

