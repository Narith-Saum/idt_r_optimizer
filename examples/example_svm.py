"""
Example 1: SVM Hyperparameter Tuning with Digits Dataset

Demonstrates IDT-R optimization for Support Vector Machine hyperparameters
using the scikit-learn digits dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from idt_r_optimizer import IDTROptimizer
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
import numpy as np


def main():
    print("\n" + "="*70)
    print("IDT-R Optimizer - Example 1: SVM Hyperparameter Tuning")
    print("="*70)

    # Load dataset
    print("\n[DATA] Loading Digits dataset...")
    X, y = load_digits(return_X_y=True)
    print(f"   Shape: {X.shape}")
    print(f"   Classes: {len(np.unique(y))}")

    # Define search space
    search_space = {
        "C": (0.1, 100.0),                    # Regularization parameter
        "kernel": ["linear", "poly", "rbf"],  # Kernel type
        "gamma": (0.0001, 1.0),               # Kernel coefficient
        "degree": (2, 5),                     # Polynomial degree
    }

    print("\n[SEARCH] Search Space:")
    for param, spec in search_space.items():
        print(f"   {param}: {spec}")

    # Define objective function
    def objective(params):
        """
        Objective: Maximize cross-validation accuracy on Iris dataset.
        """
        try:
            # Create SVM model
            model = SVC(random_state=42, **params)

            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            score = cv_scores.mean()

            return score
        except Exception as e:
            print(f"Error: {e}")
            return 0.0

    # Create optimizer
    print("\n[OPTIMIZER] Creating IDT-R Optimizer...")
    optimizer = IDTROptimizer(
        search_space=search_space,
        max_iterations=15,
        n_random_init=5,
        n_top_leaves=3,
        n_samples_per_leaf=2,
        tree_max_depth=4,
        verbose=True,
        maximize=True,
        seed=42,
    )

    # Run optimization
    print("\n[RUNNING] Running optimization...")
    best_params, best_score = optimizer.optimize(objective)

    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
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

    print(f"\n[SUMMARY] Optimization Summary:")
    print(f"   Total Evaluations: {summary['n_evaluations']}")
    print(f"   Best Score: {summary['best_score']:.6f}")
    print(f"   Mean Score: {summary['mean_score']:.6f}")
    print(f"   Std Dev: {summary['std_score']:.6f}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
