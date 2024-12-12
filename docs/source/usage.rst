Usage
=====

BioNeuralNet provides a suite of tools to integrate omics data with neural network embeddings. Below are some basic usage examples to help you get started.

.. toctree::
   :maxdepth: 2
   :caption: Usage Examples:

   ../examples/usage_examples
```

**Explanation:**

- **`toctree`**: Links to the `usage_examples.py` in the `examples/` directory, providing practical examples.

---

### **c. `docs/source/api_reference.rst`**

```rst
API Reference
=============

The API Reference provides detailed documentation for each module, class, and function within BioNeuralNet.

.. autosummary::
    :toctree: _autosummary
    :recursive:

    bioneuralnet.graph_generation.SmCCNet
    bioneuralnet.graph_generation.WGCNA
    bioneuralnet.network_embedding.GNNEmbedding
    bioneuralnet.network_embedding.Node2VecEmbedding
    bioneuralnet.subject_representation.SubjectRepresentationEmbedding
    bioneuralnet.utils.file_helpers.find_files
    bioneuralnet.utils.path_utils.validate_paths
```

**Explanation:**

- **`autosummary`**: Automatically generates summary tables and links to detailed documentation for each listed component.
- **`recursive`**: Ensures that all submodules are included.

**Note:** Ensure that `autosummary_generate = True` is set in `conf.py` to enable automatic generation of summary tables.

---