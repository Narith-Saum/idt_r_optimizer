"""
Main IDT-R Optimizer class.

Implements the Iterative Decision Tree - Random (IDT-R) hyperparameter optimization algorithm.
"""

from typing import Callable, Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .search_space import SearchSpace
from .history import OptimizationHistory
from .tree_utils import TreeNodeExtractor, LeafNode
from .sampler import LeafSampler
from .utils import (
    remove_duplicate_params,
    filter_new_params,
    format_params_for_display,
)


class IDTROptimizer:
    """
    Iterative Decision Tree - Random (IDT-R) Optimizer.

    A hyperparameter optimization algorithm that combines:
    1. Decision Tree surrogate model for learning objective landscape
    2. Direct node extraction to identify promising regions
    3. Random sampling within leaf node bounds
    4. Iterative refinement

    Parameters:
    -----------
    search_space : Dict[str, Any]
        Dictionary defining parameter bounds. Example:
        {
            "learning_rate": (0.001, 0.1),
            "n_estimators": (10, 500),
            "criterion": ["gini", "entropy"],
        }

    max_iterations : int
        Maximum number of optimization iterations

    n_random_init : int
        Number of initial random evaluations

    n_top_leaves : int
        Number of best leaves to sample from per iteration

    n_samples_per_leaf : int
        Number of random samples to generate per selected leaf

    tree_max_depth : int
        Maximum depth of surrogate decision tree

    verbose : bool
        Whether to print iteration progress

    maximize : bool
        Whether to maximize (True) or minimize (False) objective

    seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        max_iterations: int = 20,
        n_random_init: int = 5,
        n_top_leaves: int = 3,
        n_samples_per_leaf: int = 2,
        tree_max_depth: int = 5,
        verbose: bool = True,
        maximize: bool = True,
        seed: Optional[int] = None,
    ):
        self.search_space = SearchSpace(search_space)
        self.max_iterations = max_iterations
        self.n_random_init = n_random_init
        self.n_top_leaves = n_top_leaves
        self.n_samples_per_leaf = n_samples_per_leaf
        self.tree_max_depth = tree_max_depth
        self.verbose = verbose
        self.maximize = maximize
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.history = OptimizationHistory(maximize=maximize)
        self.tree: Optional[DecisionTreeRegressor] = None
        self._iteration = 0

    def optimize(self, objective_function: Callable[[Dict[str, Any]], float]) -> Tuple[Dict[str, Any], float]:
        """
        Run IDT-R optimization.

        Parameters:
        -----------
        objective_function : Callable
            Function that takes a parameter dict and returns a float score

        Returns:
        --------
        Tuple[Dict[str, Any], float]
            (best_params, best_score)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("IDT-R Optimizer - Starting Optimization")
            print(f"{'='*70}")
            print(f"Search Space: {self.search_space}")
            print(f"Initial Random Samples: {self.n_random_init}")
            print(f"Max Iterations: {self.max_iterations}")
            print(f"Top Leaves: {self.n_top_leaves}")
            print(f"Samples per Leaf: {self.n_samples_per_leaf}")
            print(f"{'='*70}\n")

        # Phase 1: Initial random exploration
        self._phase_random_init(objective_function)

        # Phase 2: Iterative optimization
        for iteration in range(self.max_iterations):
            self._iteration = iteration + 1

            if self.verbose:
                print(f"\n--- Iteration {self._iteration} ---")

            # Train surrogate model
            self._train_surrogate()

            # Extract leaf nodes
            leaves = self._extract_leaves()

            if self.verbose:
                print(f"Surrogate model trained. Found {len(leaves)} leaf nodes.")

            if not leaves:
                if self.verbose:
                    print("No leaves found. Stopping optimization.")
                break

            # Get top leaves
            top_leaves = self._select_top_leaves(leaves)

            if self.verbose:
                print(f"Selected top {len(top_leaves)} leaves for sampling.")

            # Sample from top leaves
            new_candidates = self._sample_from_leaves(top_leaves)

            if not new_candidates:
                if self.verbose:
                    print("No new candidates generated. Stopping optimization.")
                break

            # Remove duplicates
            new_candidates = remove_duplicate_params(new_candidates)

            # Filter already evaluated
            scored_params = self.history.get_all_params()
            new_candidates = filter_new_params(new_candidates, scored_params)

            if not new_candidates:
                if self.verbose:
                    print("All candidates already evaluated. Stopping optimization.")
                break

            # Evaluate new candidates
            self._evaluate_candidates(objective_function, new_candidates)

        # Final results
        best_params = self.history.get_best_params()
        best_score = self.history.get_best_score()

        if self.verbose:
            print(f"\n{'='*70}")
            print("Optimization Complete")
            print(f"{'='*70}")
            print(f"Total Evaluations: {self.history.size()}")
            print(f"Best Score: {best_score:.6f}")
            print(f"Best Parameters: {format_params_for_display(best_params)}")
            print(f"{'='*70}\n")

        return best_params, best_score

    def _phase_random_init(self, objective_function: Callable) -> None:
        """Initial phase: random exploration."""
        if self.verbose:
            print(f"Phase 1: Random Initialization ({self.n_random_init} samples)")

        candidates = self.search_space.sample_uniform(self.n_random_init)
        self._evaluate_candidates(objective_function, candidates)

    def _train_surrogate(self) -> None:
        """Train decision tree surrogate model."""
        params = self.history.get_all_params()
        scores = self.history.get_all_scores()

        # Convert params to normalized array
        X = np.array([
            self.search_space.normalize(p) for p in params
        ], dtype=np.float32)

        y = np.array(scores, dtype=np.float32)

        # Train tree
        self.tree = DecisionTreeRegressor(
            max_depth=self.tree_max_depth,
            min_samples_leaf=1,
            random_state=self.seed,
        )
        self.tree.fit(X, y)

    def _extract_leaves(self) -> List[LeafNode]:
        """Extract leaf nodes from trained tree."""
        if self.tree is None:
            return []

        param_names = self.search_space.get_param_names()
        leaves = TreeNodeExtractor.extract_leaf_nodes(self.tree, param_names)
        return leaves

    def _select_top_leaves(self, leaves: List[LeafNode]) -> List[LeafNode]:
        """Select best performing leaves."""
        top_leaves = TreeNodeExtractor.get_top_leaves(
            leaves,
            n_top=min(self.n_top_leaves, len(leaves)),
            maximize=self.maximize,
        )
        return top_leaves

    def _sample_from_leaves(self, leaves: List[LeafNode]) -> List[Dict[str, Any]]:
        """Generate random samples from selected leaves."""
        bounds_list = [leaf.param_bounds for leaf in leaves]
        candidates = LeafSampler.sample_from_multiple_leaves(
            self.search_space,
            bounds_list,
            n_samples_per_leaf=self.n_samples_per_leaf,
        )
        return candidates

    def _evaluate_candidates(
        self,
        objective_function: Callable,
        candidates: List[Dict[str, Any]],
    ) -> None:
        """Evaluate a list of candidates and update history."""
        for params in candidates:
            try:
                score = objective_function(params)
                self.history.add_record(self._iteration, params, score)

                if self.verbose:
                    score_str = f"{score:.6f}"
                    params_str = format_params_for_display(params)
                    print(f"  Evaluated: score={score_str}, params={params_str}")

            except Exception as e:
                if self.verbose:
                    print(f"  Error evaluating {params}: {e}")

    def get_history(self) -> OptimizationHistory:
        """Return the optimization history."""
        return self.history

    def get_best(self) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Return best parameters and score found."""
        return self.history.get_best_params(), self.history.get_best_score()

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of optimization results."""
        summary = self.history.get_summary()
        summary["algorithm"] = "IDT-R"
        summary["iterations"] = self._iteration
        return summary
