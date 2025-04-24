## Preprocessing Summary

1. **Raw Data Sources**  
    - **miRNA:** `BRCA.miRseq_raw_counts.txt`  
    - **RNA:** `BRCA.mRNAseq_raw_counts.txt`  
    - **Methylation:** `BRCA.meth.by_mean.data.txt`  
    - **Clinical:** `BRCA.clin.merged.picked.txt`  
    - **PAM50 Subtypes:** `pam50_subtypes.csv`  

2. **Column-Level Filtering**  
    - Kept only primary tumor samples (barcodes ending in `-01`)  
    - Trimmed sample IDs to the first 15 characters  

3. **Orientation and Cleanup**  
    - Transposed all data to a **samples × features** format  
    - Dropped reference columns (e.g., `Composite Element REF`)  
    - Normalized sample IDs: replaced hyphens (`-`) with underscores (`_`) and lowercased all characters  

4. **Value Transformations**

    | Modality       | Input         | Transformation                                      |
    |----------------|---------------|-----------------------------------------------------|
    | **miRNA**      | Raw counts    | Min-max scaling to logit: `log2(p / (1 - p))`       |
    | **RNA**        | Raw counts    | Min-max scaling to logit: `log2(p / (1 - p))`       |
    | **Methylation**| beta-values      | Clipping to m-values: `log2(beta / (1 - β))`     |

5. **Common Sample Intersection**  
    - Identified shared sample IDs across all five tables  

6. **Splitting Large Tables**  
    - RNA and Methylation tables (over 100 MB) were split into `_part1.csv` and `_part2.csv` for efficiency  
    - These are automatically rejoined during loading  

7. **Dataset Loading Logic**  
    - When loading a table, the `DatasetLoader` looks in the following order:
        1. `{stem}_common.csv`  
        2. `{stem}_part1.csv` and `{stem}_part2.csv`  
        3. `{stem}.csv` (fallback)  
    - Ensures consistency, completeness, and performance during analysis  
