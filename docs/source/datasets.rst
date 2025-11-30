Datasets Guide
==============

BioNeuralNet ships with several built-in multi-omics datasets that can be loaded via convenience functions or through the :class:`bioneuralnet.datasets.DatasetLoader` class.

Each dataset is loaded as a collection of :class:`pandas.DataFrame` objects:

- Keys are table names (e.g., ``"rna"``, ``"mirna"``, ``"clinical"``, ``"target"``).
- Values are the corresponding data tables.

In BioNeuralNet, **rows** represent subjects/patients and **columns** represent omics features or related variables.

Feature Selection Summary
-------------------------

Because omics data are typically **high-dimensional**, most built-in datasets use a filtering step based on both:

- **ANOVA F-test**: To capture features with strong class separability under a linear model.
- **Random Forest importance**: To capture non-linear dependencies and interaction effects.

We retain features in the **intersection** of these criteria, providing a compact set that is both discriminative and predictive. For a detailed mathematical description of this procedure, see :ref:`feature-selection-details` below.

Example overlap for selected omics types:

.. list-table:: Feature overlap across selection methods
   :header-rows: 1

   * - Omics Data Type
     - ANOVA-F & Variance
     - RF & Variance
     - ANOVA-F & Random Forest (Selected)
     - All Three Agree
   * - Methylation
     - 2,092 features
     - 1,870 features
     - **2,203 features**
     - 814 features
   * - RNA
     - 2,359 features
     - 2,191 features
     - **2,500 features**
     - 1,124 features

Quick Usage
-----------

You can use either the convenience loader functions or the lower-level
:class:`DatasetLoader`:

.. code-block:: python

   from bioneuralnet.datasets import (
       load_example,
       load_monet,
       load_brca,
       load_lgg,
       load_kipan,
       load_paad,
   )

   brca = load_brca()
   print(brca.keys())
   # dict_keys(['mirna', 'target', 'clinical', 'rna', 'meth'])

   from bioneuralnet.datasets import DatasetLoader

   loader = DatasetLoader("kipan")
   print(loader.shape)
   # {'mirna': (658, 472), 'target': (658, 1), 'clinical': (658, 19),
   #  'rna': (658, 2284), 'meth': (658, 2102)}

API Summary
-----------

Convenience loaders
^^^^^^^^^^^^^^^^^^^

The following functions are provided in `bioneuralnet.datasets`:

.. code-block:: python

   from bioneuralnet.datasets import (
       load_example,
       load_monet,
       load_brca,
       load_lgg,
       load_kipan,
       load_paad,
   )

Each function returns a ``dict[str, pandas.DataFrame]`` mapping table names to loaded DataFrames:

- :func:`load_example` keys: ``"X1"``, ``"X2"``, ``"Y"``, ``"clinical"``
- :func:`load_monet` keys: ``"gene"``, ``"mirna"``, ``"phenotype"``, ``"rppa"``, ``"clinical"``
- :func:`load_brca` keys: ``"mirna"``, ``"target"``, ``"clinical"``, ``"rna"``, ``"meth"``
- :func:`load_lgg` keys: ``"mirna"``, ``"target"``, ``"clinical"``, ``"rna"``, ``"meth"``
- :func:`load_kipan` keys: ``"mirna"``, ``"target"``, ``"clinical"``, ``"rna"``, ``"meth"``
- :func:`load_paad` keys: ``"cnv"``, ``"target"``, ``"clinical"``, ``"rna"``, ``"meth"``

DatasetLoader
^^^^^^^^^^^^^

The :class:`DatasetLoader` class provides a unified interface to any of the built-in datasets:

.. code-block:: python

   from bioneuralnet.datasets import DatasetLoader

   loader = DatasetLoader("paad")
   data = loader.data
   print(loader.shape)
   # {'cnv': (177, 1035), 'target': (177, 1), 'clinical': (177, 19),
   #  'rna': (177, 1910), 'meth': (177, 1152)}

Valid ``dataset_name`` values (case-insensitive):

- ``"example"``
- ``"monet"``
- ``"brca"``
- ``"lgg"``
- ``"kipan"``
- ``"paad"``

The `.shape` property returns a mapping from table name to ``(n_rows, n_cols)`` for each loaded table.

Built-in Datasets
-----------------

Below are the built-in datasets exactly as defined in the :class:`bioneuralnet.datasets.DatasetLoader` implementation.

example
^^^^^^^

Synthetic example dataset.

Tables and shapes:

- ``X1``: ``(358, 500)``
- ``X2``: ``(358, 100)``
- ``Y``: ``(358, 1)``
- ``clinical``: ``(358, 6)``

Loaded via:

.. code-block:: python

   from bioneuralnet.datasets import load_example
   example = load_example()

monet
^^^^^

MONET multi-omics dataset included with the package.

Tables and shapes:

- ``gene``: ``(107, 5039)``
- ``mirna``: ``(107, 789)``
- ``phenotype``: ``(106, 1)``
- ``rppa``: ``(107, 175)``
- ``clinical``: ``(107, 5)``

Loaded via:

.. code-block:: python

   from bioneuralnet.datasets import load_monet
   monet = load_monet()

brca
^^^^

Breast Invasive Carcinoma (BRCA) dataset.

Tables and shapes:

- ``mirna``: ``(769, 503)``
- ``target``: ``(769, 1)``
- ``clinical``: ``(769, 103)``
- ``rna``: ``(769, 2500)``
- ``meth``: ``(769, 2203)``

Loaded via:

.. code-block:: python

   from bioneuralnet.datasets import load_brca
   brca = load_brca()

lgg
^^^

Brain Lower Grade Glioma (LGG) dataset.

Tables and shapes:

- ``mirna``: ``(511, 548)``
- ``target``: ``(511, 1)``
- ``clinical``: ``(511, 13)``
- ``rna``: ``(511, 2127)``
- ``meth``: ``(511, 1823)``

Loaded via:

.. code-block:: python

   from bioneuralnet.datasets import load_lgg
   lgg = load_lgg()

paad
^^^^

Pancreatic Adenocarcinoma (PAAD) dataset.

Tables and shapes:

- ``cnv``: ``(177, 1035)``
- ``target``: ``(177, 1)``
- ``clinical``: ``(177, 19)``
- ``rna``: ``(177, 1910)``
- ``meth``: ``(177, 1152)``

Loaded via:

.. code-block:: python

   from bioneuralnet.datasets import load_paad
   paad = load_paad()

kipan
^^^^^

Pan-kidney cohort (KIPAN: KICH + KIRC + KIRP).

Tables and shapes:

- ``mirna``: ``(658, 472)``
- ``target``: ``(658, 1)``
- ``clinical``: ``(658, 19)``
- ``rna``: ``(658, 2284)``
- ``meth``: ``(658, 2102)``

Loaded via:

.. code-block:: python

   from bioneuralnet.datasets import load_kipan
   kipan = load_kipan()


.. _feature-selection-details:

Feature Selection
-----------------

To reduce the dimensionality of the high-feature omics datasets, we employed two complementary criteria: statistical class separation via ANOVA and model-based predictive utility via Random Forests. The final feature set was obtained from the intersection of these two methods, ensuring that retained variables exhibit both strong discriminative structure and demonstrable predictive value, a more conservative and robust strategy than relying on either criterion alone.

**ANOVA F-test.**
For each feature :math:`x_j`, class separability was quantified using a one-way ANOVA.

Let :math:`\bar{x}_{k,j}` denote the class means, :math:`\bar{x}_j` the global mean, and :math:`n_k` the class sizes. The between-class and within-class mean squares are

.. math::

   MS_{\text{between},j}
   \;=\;
   \frac{1}{K-1}
   \sum_{k=1}^{K} n_k\bigl(\bar{x}_{k,j}-\bar{x}_j\bigr)^2,

.. math::

   MS_{\text{within},j}
   \;=\;
   \frac{1}{n-K}
   \sum_{k=1}^{K}
   \sum_{x_{ij}\in X_{k,j}}
   \bigl(x_{ij}-\bar{x}_{k,j}\bigr)^2.

The ANOVA statistic is then

.. math::

   F_j \;=\; \frac{MS_{\text{between},j}}{MS_{\text{within},j}}.

Features were ranked by :math:`F_j` after Benjamini-Hochberg FDR correction.

**Random Forest Importance.**
Predictive relevance was quantified using the mean decrease in Gini impurity.
A node :math:`t` with class proportions :math:`p_k(t)` has impurity

.. math::

   G(t) \;=\; 1 - \sum_{k=1}^K p_k(t)^2.

A split on feature :math:`j` produces an impurity reduction

.. math::

   \Delta G(j,t)
   \;=\;
   G(t)
   -
   \Bigl(
   \frac{N_{t_L}}{N_t} G(t_L)
   +
   \frac{N_{t_R}}{N_t} G(t_R)
   \Bigr),

and the forest-level importance is the average of :math:`\Delta G` over all trees:

.. math::

   I(j)
   \;=\;
   \frac{1}{B}
   \sum_{b=1}^{B}
   \sum_{t:\,\mathrm{feat}(t)=j}
   \Delta G(j,t).

**Consensus Selection.**
For methylation and RNA, the top :math:`6{,}000` ANOVA and Random Forest features were extracted, and their intersection was taken:

.. math::

   \mathcal{S}_{\text{final}}
   \;=\;
   \mathcal{S}_{\mathrm{ANOVA}}
   \,\cap\,
   \mathcal{S}_{\mathrm{RF}}.

This approach retains features that are simultaneously statistically discriminative and useful to a non-linear classifier, yielding a stable and biologically meaningful subset. The lower-dimensional miRNA panel (472 features) was included in full.
