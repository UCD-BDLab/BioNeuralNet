Hybrid Example 3: SmCCNet + DPMON
======================================

This tutorial demonstrates how to first generate a network using SmCCNet and then run DPMON for disease prediction.

.. literalinclude:: ../examples/hybrid_example_3.py
   :language: python
   :caption: Hybrid Example 3: SmCCNet + DPMON

**Step-by-Step Guide:**

1. **Graph Generation using SmCCNet**

   - **SmCCNet**: Generates an adjacency matrix based on multi-omics data.
   - **SmCCNet.run()**: Executes the pipeline and returns the adjacency matrix.

2. **Disease Prediction using DPMON**

   - **DPMON**: Utilizes the generated adjacency matrix along with omics and clinical data to predict disease phenotypes.
   - **DPMON.run()**: Executes the prediction pipeline and returns the predictions.

3. **Running the Workflow**

   Execute the script to perform all steps sequentially.

   ```bash
   python examples/hybrid_example_dpmon.py
   ```
4. **Results**

   Upon successful execution, the predictions will be saved in dpmon_predictions.csv.

   Notes:

   - Ensure that all input files are correctly placed in the input/ directory.
   - Review the output files to understand the results of each step.
