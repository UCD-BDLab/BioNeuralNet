Utils
=====

The utils module provides a set of utility functions that support data preprocessing, filtering, conversion, and logging in BioneuralNet. These functions help to clean and prepare omics data and network representations before further analysis.

Overview
--------

Filtering Functions
-------------------
- :func:`bioneuralnet.utils.plot_variance_distribution`
- :func:`bioneuralnet.utils.remove_variance`` removes columns from a DataFrame that have variance below a given threshold.
- :func:`bioneuralnet.utils.remove_fraction`` filters out columns with a high fraction of zero values.
- :func:`bioneuralnet.utils.network_remove_low_variance`` removes rows and columns in an adjacency matrix that exhibit low variance.
- :func:`bioneuralnet.utils.network_remove_high_zero_fraction`` removes rows and columns from an adjacency matrix based on a zero-fraction threshold.
- :func:`bioneuralnet.utils.network_filter`` applies either variance or zero-fraction filtering to an adjacency matrix.
- :func:`bioneuralnet.utils.omics_data_filter`` combines variance and zero-fraction filtering to clean omics data.

RData Conversion
----------------

- :func:`bioneuralnet.utils.rdata_to_df`` converts an RData file to a CSV file and loads it into a pandas DataFrame.

Logging
-------

- :func:`bioneuralnet.utils.get_logger`` configures and returns a logger that writes to a file named ``bioneuralnet.log`` at the project root. This function is used throughout BioneuralNet to record progress and error messages.

Example Usage
-------------

The following example demonstrates how to filter omics data and set up logging:

   .. code-block:: python

      import pandas as pd
      from bioneuralnet.utils.variance import omics_data_filter
      from bioneuralnet.utils.logger import get_logger
      logger = get_logger(__name__)

      omics = pd.read_csv('omics_data.csv')
      filtered_omics = omics_data_filter(omics, variance_threshold=1e-6, zero_frac_threshold=0.95)
      print("Filtered data shape:", filtered_omics.shape)

      logger.info("Filtering completed successfully.")

Further Information
-------------------

For more details on each function and its parameters, please refer to the inline documentation in the source code. Our GitHub repository is available from the index page.
