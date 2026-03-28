"""
Example 2: Random Forest Hyperparameter Tuning with Digits Dataset

Demonstrates IDT-R optimization for Random Forest hyperparameters
using the scikit-learn digits dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from idt_r_optimizer import IDTROptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
import numpy as np
import time


def main():
    print("\n" + "="*70)
    print("IDT-R Optimizer - Example 2: Random Forest Hyperparameter Tuning")
    print("="*70)

    # Load dataset
    print("\n[DATA] Loading Digits dataset...")
    X, y = load_digits(return_X_y=True)
    print(f"   Shape: {X.shape}")
    print(f"   Classes: {len(np.unique(y))}")

    # Define search space
    search_space = {
        "n_estimators": (10, 200),           # Number of trees
        "max_depth": (3, 30),                # Maximum tree depth
        "min_samples_split": (2, 15),        # Minimum samples to split
        "min_samples_leaf": (1, 8),          # Minimum samples in leaf
        "max_features": ["sqrt", "log2"],    # Features to consider per split
        "criterion": ["gini", "entropy"],    # Split criterion
    }

    print("\n[SEARCH] Search Space:")
    for param, spec in search_space.items():
        if isinstance(spec, list):
            print(f"   {param}: {spec}")
        else:
            print(f"   {param}: {spec}")

    # Define objective function
    def objective(params):
        """
        Objective: Maximize cross-validation accuracy.
        """
        try:
            model = RandomForestClassifier(
                n_jobs=-1,
                random_state=42,
                **params
            )

            cv_scores = cross_val_score(
                model, X, y, cv=3, scoring="accuracy", n_jobs=-1
            )
            score = cv_scores.mean()

            return score
        except Exception as e:
            print(f"Error: {e}")
            return 0.0

    # Run IDT-R optimization
    print("\n[OPTIMIZER] Starting IDT-R Optimization...")
    start_time = time.time()

    optimizer = IDTROptimizer(
        search_space=search_space,
        max_iterations=20,
        n_random_init=10,
        n_top_leaves=3,
        n_samples_per_leaf=3,
        tree_max_depth=5,
        verbose=True,
        maximize=True,
        seed=42,
    )

    best_params, best_score = optimizer.optimize(objective)
    idt_r_time = time.time() - start_time

    # Display results
    print("\n" + "="*70)
    print("IDT-R OPTIMIZATION RESULTS")
    print("="*70)
    print(f"[OK] Best Score: {best_score:.6f}")
    print(f"[OK] Best Parameters:")
    for param, value in sorted(best_params.items()):
        if isinstance(value, float):
            print(f"   {param}: {value:.6f}")
        else:
            print(f"   {param}: {value}")

    # Show history
    history = optimizer.get_history()
    summary = history.get_summary()

    print(f"\n[SUMMARY] IDT-R Summary:")
    print(f"   Total Evaluations: {summary['n_evaluations']}")
    print(f"   Best Score: {summary['best_score']:.6f}")
    print(f"   Mean Score: {summary['mean_score']:.6f}")
    print(f"   Std Dev: {summary['std_score']:.6f}")
    print(f"   Time Elapsed: {idt_r_time:.2f}s")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
