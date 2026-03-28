"""
IDT-R Optimizer: Iterative Decision Tree - Random Hyperparameter Optimization

A professional, production-ready hyperparameter optimization library based on
the IDT-R algorithm combining decision tree surrogate models with intelligent
leaf-based random sampling.
"""

from .optimizer import IDTROptimizer
from .search_space import SearchSpace, ContinuousVariable, DiscreteVariable, CategoricalVariable
from .history import OptimizationHistory, EvaluationRecord
from .tree_utils import TreeNodeExtractor, LeafNode
from .sampler import LeafSampler

__version__ = "0.1.0"
__author__ = "IDT-R Development Team"

__all__ = [
    "IDTROptimizer",
    "SearchSpace",
    "ContinuousVariable",
    "DiscreteVariable",
    "CategoricalVariable",
    "OptimizationHistory",
    "EvaluationRecord",
    "TreeNodeExtractor",
    "LeafNode",
    "LeafSampler",
]
