"""
Baseline (non-graph) models for PD sample-level prediction.

These models serve as baselines to compare against GNN performance.
"""

from .baseline_model import BaselineModel, train_baseline

__all__ = ["BaselineModel", "train_baseline"]
