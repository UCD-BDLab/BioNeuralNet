# This is a readme for the parkinson's disease graph mining project

# Defining the objective:
-  We want to find biologically meaningful motifs which manifest in small and recurring molecular interaction patterns, across multi-omics brain networks (gene, methylation, protein layers) in Parkinson's Disease.

- Each motif might represent:
    - A dysregulated signaling cascade
    - A shared transcriptional control structure
    - A conserved multi-omic regulatory pattern across brain regions

# Data Integration

1) Data Acquisition:
    - AMP-PD: RNA-seq, DNA methylation, proteomics across cortical/subcortical regions.

    - PsychENCODE / GEO: Transcriptomics and methylomics.

    - STRING / BioGRID: Protein-protein interaction priors.

    - KEGG / Reactome: For downstream motif validation.

2) Preprocessing:
    - Normalize and batch-correct per omic layer
    - Map all molecular entities to common gene identifiers
    - Create matrices Xrna, Xmeth, Xprot (samples x features)

3) Graph Construction:
- For each brain region r:
    - Build omic-specific weighted graphs
    - Combine via SmCCNet into a unified multi-layer graph:
        - Nodes = genes/proteins/CpGs
        - Edges = correlation strength or biological prior
        - Node features = omic embeddings

# Embedding

1) GNN
- Use GCNm GAT or GraphSAGE to learn embeddings, capturing each gene's multi-omic context

2) Training Objectives
- Supervised: PD vs Control Classification
- Unsupervised: Graph autoencoder reconstruction loss
- Contrastive: Preserve similarity between omic layers (RNA vs protein)

# Graph Mining

1) Community Detection
- Leiden Algorithm

2) Motif Enumeration
- Extract k-node subgraphs and use gSpan or FSG (Frequent Subgraph Mining)

3) Significance Testing
- Generate random graphs (null models) with the same degree distribution using the configuration model
- Compute expected frequency and std
- Calculate Z-scores
- Keep motifs with certain Z-score threshold

4) Compare PD vs Control Graphs
-> Highlights motifs disrupted in disease

# Biological Interpretation
1) Annotate Motifs
- Map motif nodes to gene symbols
- Match edges to biological interactions (from STRING / Reactome)

2) Pathway Enrichment
- Use GSEApy / Enrichr against KEGG and Reactome pathways
- Identify biological processes each motif corresponds to

3) Region-level Comparison
- Compute motif overlap across: Motor Cortex - Prefrontal Cortex - Temporal Cortex, to find conserved or region-specific dysregulations
