"""
IDT-R Optimizer Package

This module re-exports the idt_r package for user convenience.
Users can import from either 'idt_r' or 'idt_r_optimizer'.
"""

from idt_r import (
    IDTROptimizer,
    SearchSpace,
    ContinuousVariable,
    DiscreteVariable,
    CategoricalVariable,
    OptimizationHistory,
    EvaluationRecord,
    TreeNodeExtractor,
    LeafNode,
    LeafSampler,
)

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
