Metrics
=======

The metrics module includes functions that perform statistical analyses and generate visualizations. Functions from the metrics module allow users to assess and explore network-based embeddings, evaluate clustering performance, and explore the tabular data and network structure.

Correlation Metrics
-------------------

- :func:`bioneuralnet.metrics.omics_correlation` function computes the Pearson correlation coefficient between the first principal component of the omics data and a phenotype. The data are standardized and reduced in dimension by PCA before correlation is computed.
- :func:`bioneuralnet.metrics.cluster_correlation` function computes the Pearson correlation for a cluster of features with a phenotype. Clusters with fewer than two features or zero variance are handled appropriately.

Visualization
-------------

The module also contains several plotting functions:

- :func:`bioneuralnet.metrics.plot_variance_distribution` plots the distribution of feature variances ...
- :func:`bioneuralnet.metrics.plot_variance_by_feature` shows the variance of each feature against its index.
- :func:`bioneuralnet.metrics.plot_performance` compares performance metrics between raw and enhanced omics data.
- :func:`bioneuralnet.metrics.plot_embeddings` visualizes high-dimensional embeddings in 2D using t-SNE.
- :func:`bioneuralnet.metrics.plot_network` displays the network graph from an adjacency matrix.
- :func:`bioneuralnet.metrics.compare_clusters` compares clusters from different methods.

Example Usage
-------------

**1. Variance Distribution**
The following example generates and plots the distribution of feature variances:

.. image:: _static/variance_distribution.png
   :align: center
   :alt: Variance Distribution Plot

.. code-block:: python

   from bioneuralnet.metrics import plot_variance_distribution

   fig = plot_variance_distribution(omics2, bins=100)

---

**2. Variance Per Feature**
Plots variance for the first 20 features.

.. image:: _static/variance_by_feature.png
   :align: center
   :alt: Variance Per Feature Plot

.. code-block:: python

   from bioneuralnet.metrics import plot_variance_by_feature

   fig2 = plot_variance_by_feature(omics2.iloc[:, 0:20])


**3. GNN Embeddings**
Training a **GNN-based embedding model** and visualizing the embeddings.

.. image:: _static/plot_embeddings.png
   :align: center
   :alt: 2D t-SNE Projection of Embeddings

.. code-block:: python

   from bioneuralnet.network_embedding import GNNEmbedding

   merged_omics = pd.concat([omics1, omics2], axis=1)

   gnn = GNNEmbedding(
       adjacency_matrix=global_network,
       omics_data=merged_omics,
       phenotype_data=phenotype,
       clinical_data=clinical,
       phenotype_col="phenotype",
   )

   gnn.fit()
   embeddings = gnn.embed(as_df=True)
   display(embeddings.head())

   from bioneuralnet.metrics import plot_embeddings

   # Using our embeddings instance, we get the necessary labels for the graph.
   node_labels = gnn._prepare_node_labels()
   embeddings_array = embeddings.values  

   embeddings_plot = plot_embeddings(embeddings_array, node_labels)


**4. Network Visualization**
This section visualizes Louvain clusters in a network format.

.. image:: _static/plot_network.png
   :align: center
   :alt: Network Visualization of Louvain Clusters

.. code-block:: python

   from bioneuralnet.metrics import plot_network
   from bioneuralnet.metrics import louvain_to_adjacency

   cluster1 = louvain_clusters[0]
   cluster2 = louvain_clusters[1]

   # Convert Louvain clusters into adjacency matrices
   louvain_adj1 = louvain_to_adjacency(cluster1)
   louvain_adj2 = louvain_to_adjacency(cluster2)

   # Plot using the converted adjacency matrices
   cluster1_mapping = plot_network(louvain_adj1, weight_threshold=0.12, show_labels=True, show_edge_weights=False)
   display(cluster1_mapping.head())

   cluster2_mapping = plot_network(louvain_adj2, weight_threshold=0.12, show_labels=True, show_edge_weights=False)
   display(cluster2_mapping.head())


Further Information
-------------------

For more details on each function and its parameters, please refer to the inline documentation in the source code. Our GitHub repository is available from the index page.
