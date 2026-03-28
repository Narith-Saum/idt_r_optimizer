"""
Sampling strategies for IDT-R optimizer.

Core functionality:
- Random sampling within leaf node bounds
- Respecting variable types (continuous, discrete, categorical)
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from .search_space import SearchSpace


class LeafSampler:
    """Samples hyperparameters from within leaf node bounds."""

    @staticmethod
    def sample_from_leaf_bounds(
        search_space: SearchSpace,
        leaf_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate random samples from within a leaf's hyperparameter bounds.
        
        Parameters:
        -----------
        search_space : SearchSpace
            Search space definition
        leaf_bounds : Dict[str, Tuple[float, float]]
            Normalized bounds [0, 1] for each parameter in this leaf
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of parameter dictionaries sampled within the leaf
        """
        samples = []
        for _ in range(n_samples):
            normalized = np.zeros(len(search_space.get_param_names()))
            param_names = search_space.get_param_names()

            for i, param_name in enumerate(param_names):
                if param_name in leaf_bounds:
                    lower, upper = leaf_bounds[param_name]
                else:
                    # If not specified in bounds, use full range
                    lower, upper = 0.0, 1.0

                # Sample uniformly within bounds
                normalized[i] = np.random.uniform(lower, upper)

            # Denormalize to actual parameter values
            sample = search_space.denormalize(normalized)
            samples.append(sample)

        return samples

    @staticmethod
    def sample_from_multiple_leaves(
        search_space: SearchSpace,
        leaf_bounds_list: List[Dict[str, Tuple[float, float]]],
        n_samples_per_leaf: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Sample from multiple leaf regions.
        
        Parameters:
        -----------
        search_space : SearchSpace
            Search space definition
        leaf_bounds_list : List[Dict[str, Tuple[float, float]]]
            List of leaf bounds
        n_samples_per_leaf : int
            Number of samples per leaf
            
        Returns:
        --------
        List[Dict[str, Any]]
            Combined list of samples from all leaves
        """
        all_samples = []
        for leaf_bounds in leaf_bounds_list:
            samples = LeafSampler.sample_from_leaf_bounds(
                search_space,
                leaf_bounds,
                n_samples=n_samples_per_leaf,
            )
            all_samples.extend(samples)
        return all_samples

    @staticmethod
    def sample_uniform_global(
        search_space: SearchSpace,
        n_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate random samples from entire search space (baseline comparison).
        
        Parameters:
        -----------
        search_space : SearchSpace
            Search space definition
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of parameter dictionaries sampled uniformly
        """
        return search_space.sample_uniform(n_samples)
