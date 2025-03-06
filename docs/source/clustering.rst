Correlated Clustering
=====================

BioNeuralNet includes internal modules for performing **correlated clustering** on complex networks.
These methods extend traditional community detection by integrating **phenotype correlation**, allowing users to extract **biologically relevant, phenotype-associated modules** from any network.

paper link: https://doi.org/10.3389/fdata.2022.894632 


Overview
--------

Our framework supports three key **correlated clustering** approaches:

- **Correlated PageRank**:

  - A **modified PageRank algorithm** that prioritizes nodes based on their correlation with an external phenotype.
  
  - The **personalization vector** is computed using phenotype correlation, ensuring that **biologically significant nodes receive more influence**.
  
  - This method is ideal for **identifying high-impact nodes** within a given network.

- **Correlated Louvain**:

  - An adaptation of the **Louvain community detection algorithm**, modified to optimize for **both network modularity and phenotype correlation**.
  - The objective function for community detection is given by:

    .. math::

       Q^* = k_L \cdot Q + (1 - k_L) \cdot \overline{\lvert \rho \rvert},

    where:

      - :math:`Q` is the standard **Newman-Girvan modularity**, defined as:

        .. math::

           Q = \frac{1}{2m} \sum_{i,j} \bigl(A_{ij} - \frac{k_i k_j}{2m} \bigr) \delta(c_i, c_j),

        where :math:`A_{ij}` represents the adjacency matrix, :math:`k_i` and :math:`k_j` are node degrees, and :math:`\delta(c_i, c_j)` indicates whether nodes belong to the same community.
      - :math:`\overline{\lvert \rho \rvert}` is the **mean absolute Pearson correlation** between the **first principal component (PC1) of the subgraph's features** and the phenotype.
      - :math:`k_L` is a user-defined weight (e.g., :math:`k_L = 0.2`), balancing **network modularity and phenotype correlation**.

  - This method **detects communities** that are both **structurally cohesive and strongly associated with phenotype**.

- **Hybrid Louvain**:

  - A **refinement approach** that combines **Correlated Louvain** and **Correlated PageRank** in an iterative process.
 
  - The key steps are:

    1. **Initial Community Detection**:

       - The **input network (adjacency matrix)** is clustered using **Correlated Louvain**.
       - This identifies **initial phenotype-associated modules**.

    2. **Iterative Refinement with Correlated PageRank**:

       - In each iteration:

         - The **most correlated module** is **expanded** based on Correlated PageRank.
         - The refined network is **re-clustered using Correlated Louvain**.
         - This process continues **until convergence**.

    3. **Final Cluster Extraction**:

       - The final **phenotype-optimized modules** are extracted and returned.
       - The quality of the clustering is measured using **both modularity and phenotype correlation metrics**.

.. figure:: _static/hybrid_clustering.png
   :align: center
   :alt: Overview hybrid clustering workflow

   **Hybrid Clustering**: Precedure and steps for the hybrid clustering method.


Mathematical Approach
---------------------

**Correlated PageRank:**

   - Correlated PageRank extends the traditional PageRank formulation by **biasing the random walk towards phenotype-associated nodes**.
   
   - The **ranking function** is defined as:

  .. math::

     \mathbf{r} = \alpha \cdot \mathbf{M} \mathbf{r} + (1 - \alpha) \mathbf{p},

  where:

  - :math:`\mathbf{M}` is the transition probability matrix, derived from the **normalized adjacency matrix**.
  - :math:`\mathbf{p}` is the **personalization vector**, computed using **phenotype correlation**.
  - :math:`\alpha` is the **teleportation factor** (default: :math:`\alpha = 0.85`).

- Unlike standard PageRank, which assumes a **uniform teleportation distribution**, **Correlated PageRank prioritizes phenotype-relevant nodes**.

Graphical Comparison
--------------------

Below is an illustration of **different clustering approaches** on a sample network:

.. figure:: _static/clustercorrelation.png
   :align: center
   :alt: Comparison of Correlated Clustering Methods

   **Figure 2:** Comparison between SmCCNet generated clusters and Correlated Louvain clusters

Integration with BioNeuralNet
------------------------------

Our **correlated clustering methods** seamlessly integrate into **BioNeuralNet** and can be applied to **any network represented as an adjacency matrix**.

Use cases include:

   - **Multi-Omics Networks**: Extracting **biologically relevant subgraphs** from gene expression, proteomics, or metabolomics data.
   - **Brain Connectivity Graphs**: Identifying **functional modules associated with neurological disorders**.
   - **Social & Disease Networks**: Detecting **community structures in epidemiology and patient networks**.

Our framework supports:

   - **Graph Neural Network Embedding**: Training GNNs on **phenotype-optimized clusters**.
   
   - **Predictive Biomarker Discovery**: Identifying key **features associated with disease outcomes**.
   
   - **Customizable Modularity Optimization**: Allowing users to **adjust the trade-off between structure and phenotype correlation**.

Notes for Users
---------------

1. **Input Requirements**:

   - Any **graph-based dataset** can be used as input, provided as an **adjacency matrix**.
   
   - Phenotype data should be supplied in **numerical format** (e.g., disease severity scores, expression levels).

2. **Cluster Comparison**:

   - **Correlated Louvain extracts phenotype-associated modules.**
   
   - **Hybrid Louvain iteratively refines clusters using Correlated PageRank.**
   
   - Users can compare results using **modularity scores and phenotype correlation metrics**.

3. **Method Selection**:

   - **Correlated PageRank** is ideal for **ranking high-impact nodes in a phenotype-aware manner**.
   
   - **Correlated Louvain** is best for **detecting phenotype-associated communities**.
  
   - **Hybrid Louvain** provides the most refined, **biologically meaningful clusters**.

Conclusion
----------

The **correlated clustering methods** implemented in BioNeuralNet provide a **powerful, flexible framework** for extracting **highly structured, phenotype-associated modules** from any network.
By integrating **phenotype correlation directly into the clustering process**, these methods enable **more biologically relevant and disease-informative network analysis**.

Return to :doc:`../index`
