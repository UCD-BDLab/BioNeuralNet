# GEO Platform Annotation Files

## Problem
The multi-omics pipeline needs to map Affymetrix probe IDs to gene symbols to integrate RNA and proteomics data. Currently, a placeholder mapping is used which doesn't produce real gene symbols, resulting in 0 common genes between omics.

## Solution: Download Platform Annotation Files

### For GPL570 (Affymetrix Human Genome U133 Plus 2.0 Array)

Most of your RNA brain datasets use **GPL570**. To get proper probe-to-gene mapping:

1. **Download from GEO:**
   - Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570
   - Click "Download full table" or "SOFT formatted family file"
   - Save as `GPL570.annot` or `GPL570.txt` in the `datasets/` directory

2. **Or download directly:**
   ```bash
   # Using wget or curl
   wget https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&is_datatable=true&acc=GPL570&id=13162&db=GeoDb_blob123 -O datasets/GPL570.annot
   ```

3. **Expected format:**
   The annotation file should have:
   - A column with probe IDs (usually "ID" or "Probe Set ID")
   - A column with gene symbols (usually "Gene Symbol" or "Gene.Symbol")
   - Tab-separated or comma-separated format

### For GPL201 (Affymetrix Human Genome Focus Array)

One dataset (GSE20333) uses **GPL201**:
- Download from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL201
- Save as `GPL201.annot` in the `datasets/` directory

## Automatic Detection

The pipeline will automatically:
1. Extract platform IDs from GEO Series Matrix files
2. Look for annotation files in the `datasets/` directory
3. Use annotation files if found, otherwise fall back to placeholder mapping

## File Naming

Place annotation files in `PD-Notebooks/datasets/` with one of these names:
- `GPL570.annot`
- `GPL570.txt`
- `GPL570_annot.txt`
- `GPL570_full_table.txt`

## Alternative: Using R/biomaRt

If you have R installed, you can use biomaRt to create annotation files:

```r
library(biomaRt)
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
annot <- getBM(attributes = c("affy_hg_u133_plus_2", "hgnc_symbol"),
               mart = mart)
write.table(annot, "GPL570.annot", sep="\t", quote=FALSE, row.names=FALSE)
```

## Current Status

Without annotation files:
- ✅ Pipeline still works
- ✅ Separate graphs built for RNA and proteomics
- ❌ No cross-omic gene connections
- ❌ Limited multi-omics integration

With annotation files:
- ✅ Proper gene symbol mapping
- ✅ Common genes found across omics
- ✅ True multi-omics graph integration
- ✅ Cross-omic gene connections
