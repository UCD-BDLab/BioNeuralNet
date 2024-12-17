Hybrid Example 4: SmCCNet + PageRank Clustering + Visualization
==============================================================

This tutorial demonstrates a comprehensive workflow that integrates the following steps:

1. **Network Construction (SmCCNet)**:
   Generates a network from multi-omics data using `SmCCNet`. The resulting adjacency matrix represents relationships between features.

2. **PageRank-Based Clustering**:
   Uses a PageRank-based clustering method (`PageRank`) to identify meaningful sub-networks (clusters) from the constructed network.

3. **Visualization**:
   Visualizes the identified cluster using a static or dynamic visualization tool (e.g., `StaticVisualizer`).

**Step-by-Step Guide:**

1. **Setup Input Files**:
   - Ensure that your omics data (e.g., `omics_data.csv`) and phenotype data (e.g., `phenotype_data.csv`) are placed in the `input/` directory.
   - Verify that these files are properly indexed and contain the required samples/nodes.

2. **Run SmCCNet**:

   .. literalinclude:: ../examples/hybrid_example_4.py
      :language: python
      :lines: 21-31
      :caption: Running SmCCNet to generate the adjacency matrix.

   This step constructs the network from your multi-omics data and phenotype information.

3. **Run PageRank Clustering**:

   .. literalinclude:: ../examples/hybrid_example_4.py
      :language: python
      :lines: 42-65
      :caption: Running PageRank-based clustering.

   This step executes the PageRank clustering method using seed nodes you specify, identifying meaningful sub-networks.

4. **Visualization**:

   .. literalinclude:: ../examples/hybrid_example_4.py
      :language: python
      :lines: 76-90
      :caption: Visualizing the identified cluster.

   With the resulting cluster, a subgraph is extracted and visualized using a static visualization tool. The resulting image is saved to the output directory.

**Running the Example**:

To run this example:
   
.. code-block:: bash

   python examples/hybrid_example_4.py

Upon successful execution, you will find the adjacency matrix (from SmCCNet), the PageRank clustering results saved in `pagerank_output`, and a network visualization image in `visualization_output`.

**Result Interpretation**:

- **Clustering Results**: Includes cluster size, conductance, correlation, composite score, and correlation p-value.
- **Visualization**: Provides a graphical view of the identified sub-network, offering insights into feature relationships and potential biological significance.
