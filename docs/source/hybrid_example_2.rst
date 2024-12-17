Hybrid Example 2: WGCNA Workflow with GNN Embeddings
========================================================

This tutorial demonstrates how to perform a comprehensive workflow using WGCNA for graph generation, followed by GNN-based embedding generation and subject representation integration.

**Step-by-Step Guide:**

1. **Graph Generation using WGCNA**

   .. literalinclude:: ../examples/hybrid_example_2.py
      :language: python
      :caption: Hybrid Example 2: WGCNA Workflow with GNN Embeddings

2. **Understanding the Workflow**

   - **WGCNA**: Constructs a weighted correlation network from multi-omics data.
   - **GnnEmbedding**: Generates embeddings from the adjacency matrix.
   - **GraphEmbedding**: Integrates embeddings into omics data to enhance subject representations.

3. **Running the Workflow**

    Execute the script to perform all steps sequentially.

    ```bash
    python examples/hybrid_example_2.py
    ```
4. **Results**

    Upon successful execution, the enhanced omics data will be saved in the specified output directory.

    Notes:

    - Ensure that all input files are correctly placed in the input/ directory.
    - Review the output files to understand the results of each step.

    