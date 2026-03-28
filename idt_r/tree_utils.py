"""
Decision Tree utilities for IDT-R optimizer.

Core functionality:
- Extract leaf nodes from trained sklearn DecisionTree
- Determine hyperparameter bounds (intervals) for each leaf
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeRegressor


@dataclass
class LeafNode:
    """Represents a leaf node in the decision tree."""

    leaf_id: int
    param_bounds: Dict[str, Tuple[float, float]]  # Normalized [0, 1] bounds
    predicted_value: float  # Average prediction for this leaf
    n_samples: int  # Number of training samples in this leaf

    def __repr__(self) -> str:
        return (
            f"LeafNode(id={self.leaf_id}, value={self.predicted_value:.4f}, "
            f"samples={self.n_samples})"
        )


class TreeNodeExtractor:
    """Extracts leaf nodes and their bounds from sklearn DecisionTree."""

    @staticmethod
    def extract_leaf_nodes(
        tree: DecisionTreeRegressor,
        param_names: List[str],
    ) -> List[LeafNode]:
        """
        Extract all leaf nodes from a trained decision tree.
        
        Parameters:
        -----------
        tree : DecisionTreeRegressor
            Trained sklearn DecisionTreeRegressor
        param_names : List[str]
            Names of hyperparameters (must match tree feature order)
            
        Returns:
        --------
        List[LeafNode]
            List of leaf nodes with their bounds and predictions
        """
        leaves = []
        n_params = len(param_names)

        # Initialize bounds: each param gets [0, 1]
        initial_bounds = {name: [0.0, 1.0] for name in param_names}

        # Traverse tree to find leaves
        def traverse(node_id: int, bounds: Dict[str, List[float]]) -> None:
            tree_struct = tree.tree_

            # Check if leaf node
            if tree_struct.feature[node_id] == -2:  # -2 indicates leaf in sklearn
                # This is a leaf node
                leaf_value = float(tree_struct.value[node_id][0, 0])  # Average value in leaf
                n_samples = tree_struct.n_node_samples[node_id]

                # Convert bounds to tuples
                bounds_tuple = {
                    name: (bounds[name][0], bounds[name][1])
                    for name in param_names
                }

                leaf = LeafNode(
                    leaf_id=node_id,
                    param_bounds=bounds_tuple,
                    predicted_value=leaf_value,
                    n_samples=n_samples,
                )
                leaves.append(leaf)
            else:
                # Internal node - continue traversing
                feature_idx = tree_struct.feature[node_id]
                threshold = tree_struct.threshold[node_id]

                feature_name = param_names[feature_idx]

                # Left child (feature <= threshold)
                left_bounds = {name: b.copy() for name, b in bounds.items()}
                left_bounds[feature_name][1] = threshold  # Upper bound becomes threshold
                traverse(tree_struct.children_left[node_id], left_bounds)

                # Right child (feature > threshold)
                right_bounds = {name: b.copy() for name, b in bounds.items()}
                right_bounds[feature_name][0] = threshold  # Lower bound becomes threshold
                traverse(tree_struct.children_right[node_id], right_bounds)

        # Start traversal from root
        traverse(0, initial_bounds)

        return leaves

    @staticmethod
    def get_leaf_intervals_from_tree(
        tree: DecisionTreeRegressor,
        param_names: List[str],
    ) -> List[Dict[str, Tuple[float, float]]]:
        """
        Get interval bounds for each leaf in the tree (normalized space).
        
        Returns:
        --------
        List[Dict[str, Tuple[float, float]]]
            List where each element is a dict of param bounds for that leaf
        """
        leaves = TreeNodeExtractor.extract_leaf_nodes(tree, param_names)
        return [leaf.param_bounds for leaf in leaves]

    @staticmethod
    def rank_leaves_by_prediction(
        leaves: List[LeafNode],
        maximize: bool = True,
    ) -> List[LeafNode]:
        """
        Rank leaves by their predicted performance.
        
        Parameters:
        -----------
        leaves : List[LeafNode]
            Leaves to rank
        maximize : bool
            If True, sort descending (best first)
            If False, sort ascending (best first)
            
        Returns:
        --------
        List[LeafNode]
            Sorted leaves (best first)
        """
        reverse = maximize  # For descending sort
        return sorted(leaves, key=lambda leaf: leaf.predicted_value, reverse=reverse)

    @staticmethod
    def get_top_leaves(
        leaves: List[LeafNode],
        n_top: int,
        maximize: bool = True,
    ) -> List[LeafNode]:
        """
        Select top N leaves by predicted performance.
        
        Parameters:
        -----------
        leaves : List[LeafNode]
            Leaves to select from
        n_top : int
            Number of top leaves to select
        maximize : bool
            If True, select highest predictions
            If False, select lowest predictions
            
        Returns:
        --------
        List[LeafNode]
            Top N leaves with best predictions
        """
        ranked = TreeNodeExtractor.rank_leaves_by_prediction(leaves, maximize)
        return ranked[:n_top]
