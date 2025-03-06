Coming Soon: Multi-Modal Integration 
====================================

Overview
--------

At its current state, BioNeuralNet can be used to reduce the dimensionality of high-dimensional omics data, but it also offers flexibility to explore additional modalities.
In this example, we will use the **CPTAC data repository**: `https://github.com/PayneLab/cptac <https://github.com/PayneLab/cptac>`_ to quickly show an example.
The **CPTAC Data** module is a Python interface developed by the Payne Lab that provides fast access to the National Cancer Institute's Clinical Proteomic Tumor Analysis Consortium (CPTAC) data. This package delivers comprehensive multi-omics and clinical data (e.g. proteomics, genomics, transcriptomics, and clinical attributes) as native Python *pandas* DataFrames. The tool allows you to rapidly download, inspect, and export data from various cancer types-including clear cell renal cell carcinoma (CCRCC), breast, colon, ovarian, and more-without manual parsing or reformatting.

.. note::

   **NOTE:** Although the CPTAC data module is not part of the core BioNeuralNet package, it offers an extremely convenient and fast way to pull omics data in Python. This makes it a great starting point for users who want to integrate omics data into their analysis pipelines or pair these data with imaging and clinical datasets for multimodal research.

   We highly encourage you to review their documentation at `CPTAC data <https://pypi.org/project/cptac/>`_. Below, we offer a simple example of how flexible BioNeuralNet is when working with external packages.

CPTAC Example
-------------

To install the package from PyPI, run:

.. code-block:: bash

   pip install cptac

Usage Example: Clear Cell Renal Cell Carcinoma
----------------------------------------------

The following example shows how to list available cancers,  
load the CCRCC dataset, and extract multiple data types.

.. code-block:: python

   import cptac

   ccrcc = cptac.Ccrcc()

   proteomics = ccrcc.get_proteomics("bcm")
   genomics = ccrcc.get_dataframe("CNV", "bcm")
   clinical = ccrcc.get_clinical("mssm")

   proteomics.to_csv("ccrcc_output/ccrcc_proteomics_bcm.csv")
   genomics.to_csv("ccrcc_output/ccrcc_genomics_cnv_bcm.csv")
   clinical.to_csv("ccrcc_output/ccrcc_clinical_mssm.csv")

Output Examples
---------------

Exporting the data to CSV to take a closer look at the **CCRCC Data format**

.. figure:: _static/cptac_clinical.png
   :align: center
   :alt: CPTAC Clinical Data

.. figure:: _static/cptac_genomics.png
   :align: center
   :alt: CPTAC Genomics Data

.. figure:: _static/cptac_proteomics.png
   :align: center
   :alt: CPTAC Proteomics Data

Integration with BioNeuralNet
-----------------------------

Since the `get_data()` functions from `ccrcc` return a pandas DataFrame, integrating CPTAC data into BioNeuralNet is seamless because most components work with DataFrames.

.. code-block:: python

   import cptac
   from BioNeuralNet import external_tools.SmCCNet

   # Load the CCRCC dataset
   ccrcc = cptac.Ccrcc()

   # Retrieve omics and clinical data
   genomics = ccrcc.get_dataframe("CNV", "bcm")
   proteomics = ccrcc.get_proteomics("bcm")
   clinical = ccrcc.get_clinical("mssm")

   smccnet = SmCCNet(
       phenotype_df=clinical["tumor_stage_pathological"],
       omics_dfs=[genomics, proteomics],
       data_types=["Genes", "Proteins"],
   )

- **Ease of Integration:** As demonstrated in the BioNeuralNet documentation, the CPTAC Data module can be integrated as a data retrieval engine, feeding high-quality, reproducible data into advanced machine learning or network analysis pipelines.

Integration with Other Data Sources
-----------------------------------

Beyond omics data, the CPTAC Data module serves as an excellent entry point for multimodal research. For example, researchers can combine omics data obtained via this module with imaging data available from the **Cancer Imaging Archive**. This enables studies that integrate molecular and imaging information-vital for the development of comprehensive cancer diagnostics and treatment strategies.

For example, we can look at the **NCI Cancer Imaging Archive** to get additional modalities.  
Since we are analyzing **Clear Cell Renal Cell Carcinoma (CCRCC)**, we can search for `ccrcc` in the collection:  
`NCI Cancer Imaging Archive - CCRCC Collection <https://www.cancerimagingarchive.net/collection/cptac-ccrcc/>`_.  
We can then retrieve the respective images for the patients.

.. figure:: _static/ccrcc_search.png
   :align: center
   :alt: CCRCC Search in Cancer Imaging Archive

.. figure:: _static/images_download.png
   :align: center
   :alt: Image Download Process


**If you are working with another cancer type, there are many available.**

.. figure:: _static/cptac_search.png
   :align: center
   :alt: CPTAC Data Search


References
----------

- **PayneLab/cptac GitHub Repository:**  
  `CPTAC Data Module <https://pypi.org/project/cptac/>`_

- **Cancer Imaging Archive - Imaging-Omics:**  
   `CPTAC-CCRCC Collection <https://www.cancerimagingarchive.net/collection/cptac-ccrcc/>`_

- **NCI Clinical Proteomic Tumor Analysis Consortium (CPTAC) - CCRCC Collection (Version 13):**  
  National Cancer Institute, The Cancer Imaging Archive.  
  `CPTAC <https://www.cancerimagingarchive.net/collection/cptac-ccrcc/>`_

- **Edwards NJ, Oberti M, Thangudu RR, et al. (2015).**  
  *The CPTAC Data Portal: A Resource for Cancer Proteomics Research.*  
  J Proteome Res. 14(6):2707-13.
  `DOI: 10.1021/acs.jproteome.5b00340 <https://doi.org/10.1021/acs.jproteome.5b00340>`_  
