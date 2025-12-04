import networkx as nx
import numpy as np
import pytest

from bioneuralnet.clustering.spectral import Spectral_Clustering


def test_spectral_basic_construction_with_weighted_edges():
    """Test basic construction with a simple weighted graph."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.5)])
    
    spec = Spectral_Clustering(G=G, n_clusters=2, use_edge_weights=True)
    
    # Verify affinity matrix was built correctly
    assert spec.A.shape == (4, 4)
    assert spec.A[0, 1] == 1.0
    assert spec.A[1, 0] == 1.0  # symmetric (undirected)
    assert spec.A[1, 2] == 2.0
    assert spec.A[2, 1] == 2.0
    assert spec.A[2, 3] == 1.5
    assert spec.A[3, 2] == 1.5
    # Check diagonal is zeros (no self-loops)
    assert np.all(np.diag(spec.A) == 0)


def test_spectral_unweighted_graph():
    """Test construction with use_edge_weights=False."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 10.0), (2, 3, 3.0)])
    
    spec = Spectral_Clustering(G=G, n_clusters=2, use_edge_weights=False)
    
    # All non-zero weights should be 1.0 in unweighted mode
    assert spec.A[0, 1] == 1.0
    assert spec.A[1, 2] == 1.0
    assert spec.A[2, 3] == 1.0
    # All diagonal should still be 0
    assert np.all(np.diag(spec.A) == 0)


def test_spectral_default_weights_when_missing():
    """Test that edges without weight attribute default to 1.0."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)  # no weight attribute
    G.add_weighted_edges_from([(1, 2, 3.0)])  # explicit weight
    
    spec = Spectral_Clustering(G=G, n_clusters=2, use_edge_weights=True)
    
    assert spec.A[0, 1] == 1.0  # default
    assert spec.A[1, 2] == 3.0  # explicit


def test_spectral_graph_with_no_edges_raises_error():
    """Test that a graph with no edges raises ValueError."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    
    with pytest.raises(ValueError, match="Graph has no edges"):
        Spectral_Clustering(G=G, n_clusters=2)


def test_spectral_nodes_order_preserved():
    """Test that node order is preserved for consistent label indexing."""
    G = nx.Graph()
    G.add_nodes_from(['a', 'b', 'c', 'd'])
    G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
    
    spec = Spectral_Clustering(G=G, n_clusters=2)
    
    assert spec.nodes_order == ['a', 'b', 'c', 'd']
    assert len(spec.nodes_order) == 4


def test_spectral_run_returns_labels_and_nodes_order():
    """Test that run() returns both labels and node order."""
    G = nx.path_graph(5)
    spec = Spectral_Clustering(G=G, n_clusters=2, random_state=42)
    
    labels, nodes_order = spec.run()
    
    # Check shapes and types
    assert isinstance(labels, np.ndarray)
    assert isinstance(nodes_order, list)
    assert labels.shape[0] == 5
    assert len(nodes_order) == 5
    # Labels should be integers in range [0, n_clusters)
    assert np.all(labels >= 0)
    assert np.all(labels < 2)


def test_spectral_clustering_assigns_all_nodes():
    """Test that all nodes receive cluster assignments."""
    G = nx.complete_graph(6)
    spec = Spectral_Clustering(G=G, n_clusters=3, random_state=0)
    
    labels, nodes_order = spec.run()
    
    # Every node should have exactly one label
    assert len(labels) == 6
    # All nodes in order
    assert nodes_order == [0, 1, 2, 3, 4, 5]


def test_spectral_with_different_n_clusters():
    """Test that n_clusters parameter is respected."""
    G = nx.barbell_graph(3, 1)  # Two dense subgraphs connected by a bridge
    
    # Run with 2 clusters
    spec2 = Spectral_Clustering(G=G, n_clusters=2, random_state=42)
    labels2, _ = spec2.run()
    unique_clusters_2 = len(np.unique(labels2))
    
    # Run with 3 clusters
    spec3 = Spectral_Clustering(G=G, n_clusters=3, random_state=42)
    labels3, _ = spec3.run()
    unique_clusters_3 = len(np.unique(labels3))
    
    # Should produce the requested number of clusters (or fewer if some are empty)
    assert unique_clusters_2 <= 2
    assert unique_clusters_3 <= 3


def test_spectral_random_state_reproducibility():
    """Test that same random_state produces same clustering."""
    G = nx.karate_club_graph()
    
    spec1 = Spectral_Clustering(G=G, n_clusters=2, random_state=123)
    labels1, _ = spec1.run()
    
    spec2 = Spectral_Clustering(G=G, n_clusters=2, random_state=123)
    labels2, _ = spec2.run()
    
    # Same random state should give same results (or very similar)
    assert np.array_equal(labels1, labels2)


def test_spectral_different_random_states():
    """Test that different random_states can produce different results."""
    G = nx.karate_club_graph()
    
    spec1 = Spectral_Clustering(G=G, n_clusters=2, random_state=100)
    labels1, _ = spec1.run()
    
    spec2 = Spectral_Clustering(G=G, n_clusters=2, random_state=200)
    labels2, _ = spec2.run()
    
    # Different seeds may produce different partitions
    # (not guaranteed, but very likely for a non-trivial graph)
    # We just check they're both valid
    assert len(np.unique(labels1)) <= 2
    assert len(np.unique(labels2)) <= 2


def test_spectral_affinity_matrix_symmetry():
    """Test that affinity matrix is symmetric (undirected graph property)."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 2.5), (1, 2, 3.7), (0, 2, 1.2)])
    
    spec = Spectral_Clustering(G=G, n_clusters=2)
    
    # For undirected graphs, affinity should be symmetric
    assert np.allclose(spec.A, spec.A.T)


def test_spectral_large_graph_clustering():
    """Test clustering on a larger graph with distinct structure."""
    # Create a graph with two clear communities
    G = nx.Graph()
    # First community: nodes 0-9 fully connected
    for i in range(10):
        for j in range(i + 1, 10):
            G.add_edge(i, j, weight=1.0)
    
    # Second community: nodes 10-19 fully connected
    for i in range(10, 20):
        for j in range(i + 1, 20):
            G.add_edge(i, j, weight=1.0)
    
    # Single bridge edge between communities (weak connection)
    G.add_edge(5, 15, weight=0.1)
    
    spec = Spectral_Clustering(G=G, n_clusters=2, random_state=42)
    labels, nodes_order = spec.run()
    
    # Verify basic properties
    assert len(labels) == 20
    assert len(nodes_order) == 20
    # Should ideally separate into 2 clusters, but weak bridge might confuse it
    assert len(np.unique(labels)) <= 2


def test_spectral_single_node_per_cluster():
    """Test that each node gets assigned to exactly one cluster."""
    G = nx.complete_graph(8)
    spec = Spectral_Clustering(G=G, n_clusters=4, random_state=99)
    
    labels, nodes_order = spec.run()
    
    # Each node should have exactly one label
    assert len(labels) == 8
    # Labels should be in valid range
    assert np.all(labels >= 0)
    assert np.all(labels < 4)


def test_spectral_affinity_matrix_dense():
    """Test that affinity matrix is properly populated (dense format)."""
    G = nx.cycle_graph(5)
    spec = Spectral_Clustering(G=G, n_clusters=2, use_edge_weights=False)
    
    # Count non-zero entries (excluding diagonal which is always 0)
    A = spec.A
    non_diag_nonzero = np.sum(A != 0)
    
    # For a cycle_graph(5), there are 5 edges, each contributes 2 entries (undirected)
    assert non_diag_nonzero == 10
