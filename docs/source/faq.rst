Frequently Asked Questions (FAQ)
================================

**Q1: What is BioNeuralNet?**

A1: BioNeuralNet is a Python-based tool for integrating multi-omics data into network-based representations 
and simplifying them into lower-dimensional embeddings for downstream analyses like clustering, feature 
selection, and disease prediction.

**Q2: Should I use SmCCNet or WGCNA for graph generation?**

A2: SmCCNet is suited for sparse canonical correlation networks, while WGCNA focuses on 
weighted gene co-expression analysis. Your choice depends on the data type, research focus, and familiarity 
with these methods. You may also use import a exisiting network, most components are compatabile with adjency matrix as network input.

**Q3: Can I accelerate computations using GPUs?**

A3: Yes. BioNeuralNet supports CUDA-based installations for GPU acceleration. Ensure that your environment 
is set up with compatible GPU drivers, CUDA, and PyTorch configurations.

**Q4: How do I contribute to BioNeuralNet?**

A4: Contributions are welcome! Check the repository’s contributing guidelines for details on submitting issues, 
pull requests, and improving documentation or code.

**Q5: Where can I get help or report issues?**

A5: For help, bug reports, or feature requests, open an issue on the GitHub repository’s Issues page.
