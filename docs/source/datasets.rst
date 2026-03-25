Datasets Guide
==============

BioNeuralNet ships with several built-in multi-omics datasets that can be loaded via convenience functions or through the :class:`bioneuralnet.datasets.DatasetLoader` class.

Each dataset is loaded as a collection of :class:`pandas.DataFrame` objects:

- Keys are table names (e.g., ``"rna"``, ``"mirna"``, ``"clinical"``, ``"target"``).
- Values are the corresponding data tables.

In BioNeuralNet, **rows** represent subjects/patients and **columns** represent omics features or related variables.

Feature Selection Summary
-------------------------

To address high dimensionality and isolate the most informative variables, unsupervised feature selection was performed across all cohorts using **Laplacian Score** filtering.
This methylationod evaluates each feature based on its ability to preserve the local manifold structure of the data, emphasizing features that vary minimally between closely related samples.
Lower Laplacian Scores indicate higher feature importance.

The following feature counts were retained per modality across BRCA, LGG, and KIPAN cohorts:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Modality
     - Features Retained
     - Cohorts Applied
   * - DNA methylation
     - 400
     - BRCA, LGG, KIPAN
   * - mRNA
     - 200
     - BRCA, LGG, KIPAN
   * - miRNA
     - 100
     - BRCA, LGG, KIPAN

For a full list of available feature selection methylationods, see the `Preprocessing Utilities <https://bioneuralnet.readthedocs.io/en/latest/utils.html#preprocessing-utilities>`_ documentation.

Quick Usage
-----------

You can use either the convenience loader functions or the lower-level :class:`DatasetLoader`:

.. code-block:: python

   from bioneuralnet.datasets import (
       load_example,
       load_monet,
       load_brca,
       load_lgg,
       load_kipan,
   )

   brca = load_brca()
   print(brca.keys())
   # dict_keys(['mirna', 'target', 'clinical', 'rna', 'methylation'])

   from bioneuralnet.datasets import DatasetLoader

   loader = DatasetLoader("kipan")
   print(loader.shape)

API Summary
-----------

Each function returns a ``dict[str, pandas.DataFrame]`` mapping table names to loaded DataFrames:

- :func:`load_example` keys: ``"X1"``, ``"X2"``, ``"Y"``, ``"clinical"``
- :func:`load_monet` keys: ``"gene"``, ``"mirna"``, ``"phenotype"``, ``"rppa"``, ``"clinical"``
- :func:`load_brca` keys: ``"mirna"``, ``"target"``, ``"clinical"``, ``"rna"``, ``"methylation"``
- :func:`load_lgg` keys: ``"mirna"``, ``"target"``, ``"clinical"``, ``"rna"``, ``"methylation"``
- :func:`load_kipan` keys: ``"mirna"``, ``"target"``, ``"clinical"``, ``"rna"``, ``"methylation"``

Valid ``dataset_name`` values for :class:`DatasetLoader` (case-insensitive): ``"example"``, ``"monet"``, ``"brca"``, ``"lgg"``, ``"kipan"``.

Built-in Datasets
-----------------

example
^^^^^^^

Synthetic dataset for testing and demonstration.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Table
     - Shape
     - Description
   * - ``X1``
     - (358, 500)
     - Gene expression features
   * - ``X2``
     - (358, 100)
     - miRNA features
   * - ``Y``
     - (358, 1)
     - Continuous phenotype
   * - ``clinical``
     - (358, 6)
     - Clinical covariates

.. code-block:: python

   from bioneuralnet.datasets import load_example
   example = load_example()

monet
^^^^^

MONET multi-omics dataset.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Table
     - Shape
     - Description
   * - ``gene``
     - (107, 5039)
     - Gene expression
   * - ``mirna``
     - (107, 789)
     - miRNA expression
   * - ``phenotype``
     - (106, 1)
     - Phenotype labels
   * - ``rppa``
     - (107, 175)
     - Protein expression
   * - ``clinical``
     - (107, 5)
     - Clinical covariates

.. code-block:: python

   from bioneuralnet.datasets import load_monet
   monet = load_monet()

brca
^^^^

TCGA Breast Invasive Carcinoma (BRCA). Target: PAM50 subtype (5-class): LumA (n=419), LumB (n=140), Basal (n=130), Her2 (n=46), Normal (n=34).

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Stage
     - methylation
     - mRNA
     - miRNA
     - Clinical
   * - Raw (features x samples)
     - 20,107 x 885
     - 18,321 x 1,212
     - 503 x 1,189
     - 1,098 x 18
   * - Final aligned (samples x features)
     - 769 x 20,106
     - 769 x 16,757
     - 769 x 354
     - 769 x 17
   * - After Laplacian Score selection
     - 769 x 400
     - 769 x 200
     - 769 x 100
     - 769 x 17

.. code-block:: python

   from bioneuralnet.datasets import load_brca
   brca = load_brca()

lgg
^^^

TCGA Brain Lower Grade Glioma (LGG). Target: binary vital status, Alive (n=386) vs. Deceased (n=125).

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Stage
     - methylation
     - mRNA
     - miRNA
     - Clinical
   * - Raw (features x samples)
     - 20,115 x 685
     - 18,328 x 701
     - 548 x 531
     - 14 x 1,110
   * - Final aligned (samples x features)
     - 511 x 20,114
     - 511 x 18,328
     - 511 x 548
     - 511 x 13
   * - After Laplacian Score selection
     - 511 x 400
     - 511 x 200
     - 511 x 100
     - 511 x 13

.. code-block:: python

   from bioneuralnet.datasets import load_lgg
   lgg = load_lgg()

kipan
^^^^^

TCGA Pan-Kidney cohort (KIPAN: KICH + KIRC + KIRP). Target: binary cancer stage, Early (Stages I/II, n=417) vs. Late (Stages III/IV, n=216).

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Stage
     - methylation
     - mRNA
     - miRNA
     - Clinical
   * - Raw (features x samples)
     - 20,117 x 867
     - 18,272 x 1,020
     - 472 x 1,005
     - 20 x 941
   * - Final aligned (samples x features)
     - 658 x 20,116
     - 658 x 18,272
     - 658 x 472
     - 658 x 19
   * - After Laplacian Score selection
     - 633 x 400
     - 633 x 200
     - 633 x 100
     - 633 x 19

.. code-block:: python

   from bioneuralnet.datasets import load_kipan
   kipan = load_kipan()


.. _feature-selection-details:

Feature Selection
-----------------

To reduce the dimensionality of the high-feature omics datasets, unsupervised feature selection was performed using Laplacian Score filtering.
The Laplacian Score for the :math:`r`-th feature is:

.. math::

   L_r = \frac{\sum_{ij} (x_{ri} - x_{rj})^2 W_{ij}}{\text{Var}(x_r)}

Where:

- :math:`L_r` is the Laplacian Score for feature :math:`r`. Lower scores indicate higher importance.
- :math:`x_{ri}` and :math:`x_{rj}` are the standardized values of feature :math:`r` for samples :math:`i` and :math:`j`. All feature vectors undergo Z-score normalization prior to scoring.
- :math:`W_{ij}` is the edge weight between samples :math:`i` and :math:`j` in a symmetric k-nearest neighbors affinity graph. If samples :math:`i` and :math:`j` are neighbors, :math:`W_{ij} = 1`; otherwise :math:`W_{ij} = 0`.
- :math:`\text{Var}(x_r)` is the variance of feature :math:`r`, weighted by the degree matrix. This denominator ensures scale-invariant normalization, reflecting local spatial variance relative to global feature variance.

By filtering for the lowest Laplacian Scores, the optimal subsets of features were retained per cohort to maximize computational efficiency while preserving biological signals.