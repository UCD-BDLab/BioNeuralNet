"""External tools integration for BioNeuralNet.

This module provides wrappers to interface with external bioinformatics packages, streamlining the execution of R or Python-based tools within the BioNeuralNet pipeline.

Currently supported:

* **SmCCNet**: https://CRAN.R-project.org/package=SmCCNet.

**Extensibility**:

This module is designed to be extensible.
Users are encouraged to implement wrappers for additional external tools (e.g., WGCNA, MOFA) following the pattern established by SmCCNet.
"""

from .smccnet import SmCCNet

__all__ = ["SmCCNet"]
