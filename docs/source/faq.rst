Frequently Asked Questions (FAQ)
================================

**Q1: What is BioNeuralNet?**

A1: BioNeuralNet is a Python-based framework designed to integrate and analyze multi-omics data through network-based representations and lower-dimensional embeddings. It facilitates advanced analytical processes such as clustering, feature selection, disease prediction, and environmental exposure assessment, enabling researchers to derive actionable insights from complex biological datasets.

---

**Q2: What are the key components of BioNeuralNet?**

A2: BioNeuralNet comprises five core components:

1. **Graph Construction**: Build multi-omics networks using methods like Weighted Gene Co-expression Network Analysis (**WGCNA**), Sparse Multiple Canonical Correlation Network (**SmCCNet**), or import existing networks.
2. **Graph Clustering**: Identify functional modules and communities using hierarchical clustering, PageRank, or Louvain clustering algorithms.
3. **Network Embedding**: Generate embeddings with **Graph Neural Networks (GNNs)** or **Node2Vec**, simplifying high-dimensional data into lower-dimensional representations.
4. **Subject Representation**: Integrate embeddings into omics data to enrich subject-level features, enhancing the dataset for downstream analyses.
5. **Downstream Tasks**: Perform advanced analyses like disease prediction using network information. Seamlessly integrate your own downstream tasks by leveraging existing components.

---

**Q3: Can I request new components?**

A3: Yes! Please reach out, and we will gladly work with you to integrate new components into BioNeuralNet. Our goal is to facilitate multi-omics research by providing a flexible and extensible framework, and integrating new custom components is a key part of that.

---

**Q4: How do I install BioNeuralNet and its dependencies?**

A4: For detailed installation instructions, follow these steps: :doc:`installation`.

---

**Q5: Can I accelerate computations using GPUs?**

A5: Yes, BioNeuralNet supports CUDA-based installations for GPU acceleration, significantly enhancing computation speeds for large datasets and complex models. To enable GPU acceleration:

1. **Ensure GPU Compatibility:**
   - Your system must have a compatible NVIDIA GPU with the latest drivers installed.
   
2. **Install CUDA and cuDNN:**
   - Follow the `official NVIDIA installation guide <https://developer.nvidia.com/cuda-downloads>`_ to install CUDA and cuDNN.
   
3. **Run the Installation Script:**
   - During the `fast-install.py` setup, select the GPU-acceleration option by choosing '2' when prompted.

**Note:** Proper GPU setup is crucial for optimal performance. Verify CUDA installation by running:

.. code-block:: bash

   nvcc --version

---

**Q6: How do I contribute to BioNeuralNet?**

A6: Contributions to BioNeuralNet are highly appreciated! To contribute:

1. **Fork the Repository:**

   .. code-block:: bash

      git clone https://github.com/UCD-BDLab/BioNeuralNet.git
      cd BioNeuralNet

2. **Create a Feature Branch:**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

3. **Install Development Dependencies:**

   .. code-block:: bash

      ./setup.sh

4. **Make Your Changes:**
   - Implement new features.
   - Add or update tests.
   - Document your changes.

5. **Run Tests:**

   .. code-block:: bash

      pytest

6. **Commit Changes:**

   .. code-block:: bash

      git add .
      git commit -m "Add feature XYZ"

7. **Push and Open a Pull Request:**

   .. code-block:: bash

      git push origin feature/your-feature-name

For detailed guidelines, `BioNeuralNet README.md <https://github.com/UCD-BDLab/BioNeuralNet/blob/main/README.md>`_.

---

**Q7: Where can I find BioNeuralNet’s documentation and examples?**

A7: You are already here! For additional questions, please reach out to us at **vicente.ramos@ucdenver.edu**.

---

**Q8: How do I report issues or request features for BioNeuralNet?**

A8: For help, bug reports, or feature requests, please open an issue on the `BioNeuralNet GitHub repository's Issues page <https://github.com/UCD-BDLab/BioNeuralNet/issues>`_. Before submitting, please ensure that your issue hasn’t already been reported.

**Steps to Report an Issue:**
1. Navigate to the `Issues <https://github.com/UCD-BDLab/BioNeuralNet/issues>`_ tab.
2. Click on **"New issue"**.
3. Choose between a **Bug report** or **Feature request** template.
4. Provide a clear and descriptive title.
5. Fill in the necessary details, including steps to reproduce (for bugs) or a detailed description (for features).
6. Submit the issue.

Our team will review your submission and respond accordingly.

---

**Q9: What license is BioNeuralNet distributed under?**

A9: BioNeuralNet is distributed under the `MIT License <https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE>`_. This permissive license allows you to freely use, modify, and distribute the software, provided that the original copyright notice and permission notice are included in all copies or substantial portions of the software.

---

**Q10: Who are the contributors and maintainers of BioNeuralNet?**

A10: BioNeuralNet is developed and maintained by the `UCD-BDLab <https://github.com/UCD-BDLab>`_ team. We extend our gratitude to all contributors and open-source communities that have supported the development of BioNeuralNet. For a complete list of contributors, visit the `Contributors <https://github.com/UCD-BDLab/BioNeuralNet/graphs/contributors>`_ page on GitHub.

---

**Q11: How can I get involved with the UCD-BDLab?**

A11: The `UCD-BDLab <https://github.com/UCD-BDLab>`_ welcomes collaboration and participation from researchers and developers interested in bioinformatics and computational biology. To get involved:

1. **Explore Our Projects:** Visit the `UCD-BDLab GitHub repository <https://github.com/UCD-BDLab>`_ to see our current projects and initiatives.
2. **Contribute to Projects:** Fork repositories, contribute code, report issues, and propose new features.
3. **Join Our Community:** Participate in discussions, attend lab meetings, and collaborate on research endeavors.
4. **Contact Us:** Reach out via `GitHub Issues <https://github.com/UCD-BDLab/BioNeuralNet/issues>`_ or email the maintainers at **vicente.ramos@ucdenver.edu** for more information.

---

**Q12: What acknowledgments are associated with BioNeuralNet?**

A12: BioNeuralNet leverages several open-source libraries and tools that are integral to its functionality:

- `SmCCNet on CRAN <https://cran.r-project.org/package=SmCCNet>`_
- `WGCNA on CRAN <https://cran.r-project.org/package=WGCNA>`_
- `Node2Vec on GitHub <https://github.com/aditya-grover/node2vec>`_
- `PyTorch <https://pytorch.org/>`_
- `PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric>`_
- `dplyr on CRAN <https://cran.r-project.org/package=dplyr>`_
- `Pytest <https://pytest.org/>`_
- `Pre-commit <https://github.com/pre-commit/pre-commit>`_

We extend our gratitude to all contributors and open-source communities that have made BioNeuralNet possible.
