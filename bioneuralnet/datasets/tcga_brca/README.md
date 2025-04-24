# TCGA BRCA Data Preprocessing and Feature Selection

## Data Sources

- [FireBrowse BRCA cohort](http://firebrowse.org/?cohort=BRCA)
- [Download BRCA data](http://firebrowse.org/?cohort=BRCA&download_dialog=true)

## 1. Data Loading

Raw data loaded from Firehose:

```python
mirna_df = pd.read_csv("BRCA.miRseq_raw_counts.txt")
rna_df = pd.read_csv("BRCA.mRNAseq_raw_counts.txt")
meth_df = pd.read_csv("BRCA.meth.by_mean.data.txt")
clinical_df = pd.read_csv("BRCA.clin.merged.picked.txt")
pam50_df = pd.read_csv("pam50_subtypes.csv")
```

Initial data dimensions:

```
miRNA: (1078, 503)
RNA: (775, 20532)
Methylation:(783, 20106)
Clinical: (1097, 18)
PAM50: (1087,)
```

## 2. Data Preprocessing Steps

- **Tumor Filtering:** Retain samples ending with `-01` (primary tumors).

- **Sample ID Standardization:** Shorten to 15 characters, replace `-` with `_`, convert to lowercase.

- **Transpose Data:** Organize tables into (samples × features) format.

- **Drop Unnecessary Columns:** Remove columns like `Composite Element REF`.

- **Transformations:**

    - **miRNA/RNA:** Min-max scaling, then apply logit transform (`log2(p/(1-p))`).
    - **Methylation:** Clip β-values and convert to m-values (`log2(β/(1-β))`).

- **Common Samples:** Keep only samples present across all datasets (miRNA, RNA, Meth, Clinical, PAM50).

- **Large File Handling:** RNA and Meth datasets split into smaller parts, rejoined automatically when loaded.

- **Processed Files:** Saved as:
  - `BRCA_miRNA_processed.csv`
  - `BRCA_RNA_processed.csv`
  - `BRCA_Meth_processed.csv`
  - `BRCA_Clinical_Level4.csv`
  - `BRCA_PAM50.csv`

## 3. DatasetLoader Behavior

`DatasetLoader("tcga_brca")` searches and loads:

1. `{stem}_common.csv`
2. `{stem}_part1.csv`, `{stem}_part2.csv`
3. `{stem}.csv` (fallback)

This ensures efficient and reliable data loading.

## 4. Feature Selection Methods

Performed separately on Methylation and RNA datasets (top 1,000 features each):

- **Unsupervised:**
    - Variance Filter (highest variance)
    - Autoencoder Weights (largest weights)

- **Supervised:**
    - ANOVA F-test
    - RandomForest Feature Importance

### Method Overlaps (Top 1,000 Features)

**Methylation:**
```
    4-way overlap: 2 (0.2%)
    Variance & Autoencoder: 64 (6.4%)
    Variance & ANOVA: 67 (6.7%)
    Variance & RF: 54 (5.4%)
    Autoencoder & ANOVA: 50 (5.0%)
    Autoencoder & RF: 37 (3.7%)
    ANOVA & RF: 228 (22.8%)
```

**RNA:**
```
    4-way overlap: 0 (0.0%)
    Variance & Autoencoder: 54 (5.4%)
    Variance & ANOVA: 9 (0.9%)
    Variance & RF: 7 (0.7%)
    Autoencoder & ANOVA: 52 (5.2%)
    Autoencoder & RF: 58 (5.8%)
    ANOVA & RF: 328 (32.8%)
```

## 5. SmCCNet Network Generation

Using selected features (VAR, AE, ANOVA, RF):

```python
smcc = SmCCNet(
    phenotype_df=phenotype_df,
    omics_dfs=[miRNA_df, Meth_df, RNA_df],
    data_types=["mirna", "methylation", "rnaseq"],
    subSampNum=500,
    output_dir="./smccnet_var"
)
network, clusters = smcc.run()
```
Here's your documentation snippet updated with your exact `NetworkLoader` implementation at the end:


### 6. Network Loader Utility

To easily load SmCCNet outputs, use the provided `NetworkLoader` class from `network_loader.py`:

```python
from bioneuralnet.datasets.network_loader import NetworkLoader
loader = NetworkLoader()
print(loader.available_methods())
# ['brca_smccnet_ae', 'brca_smccnet_rf', 'brca_smccnet_var']

global_net = loader.load_global_network("brca_smccnet_var")
clusters = loader.load_clusters("brca_smccnet_var")
```
