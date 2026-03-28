"""
Example 3: Neural Network Hyperparameter Tuning with MNIST Dataset

Demonstrates IDT-R optimization for deep learning hyperparameters
using TensorFlow/Keras and the MNIST dataset.

Requires: tensorflow (install with: pip install tensorflow)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from idt_r_optimizer import IDTROptimizer
import numpy as np


def main():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("\n[ERROR] TensorFlow is required for this example.")
        print("Install it with: pip install tensorflow")
        return

    print("\n" + "="*70)
    print("IDT-R Optimizer - Example 3: Neural Network Hyperparameter Tuning")
    print("="*70)

    # Load and preprocess MNIST dataset
    print("\n[DATA] Loading and preprocessing MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize and flatten
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # Use subset for faster optimization
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]

    print(f"   Training Shape: {X_train.shape}")
    print(f"   Test Shape: {X_test.shape}")
    print(f"   Classes: 10 (digits 0-9)")

    # Define search space for hyperparameters
    search_space = {
        "learning_rate": (0.0001, 0.01),       # Learning rate
        "batch_size": (16, 128),               # Batch size (discrete)
        "hidden_units": (64, 512),             # Hidden layer size
        "dropout_rate": (0.1, 0.5),            # Dropout rate
        "optimizer": ["adam", "sgd", "rmsprop"],  # Optimizer type
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
        Objective: Maximize validation accuracy on MNIST.
        """
        try:
            # Extract parameters
            learning_rate = params["learning_rate"]
            batch_size = int(params["batch_size"])  # Convert to int
            hidden_units = int(params["hidden_units"])  # Convert to int
            dropout_rate = params["dropout_rate"]
            optimizer_name = params["optimizer"]

            # Create optimizer
            if optimizer_name == "adam":
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == "sgd":
                optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
            else:  # rmsprop
                optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

            # Build model
            model = keras.Sequential([
                layers.Dense(hidden_units, activation="relu", input_shape=(784,)),
                layers.Dropout(dropout_rate),
                layers.Dense(64, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(10, activation="softmax"),
            ])

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train model (fewer epochs for faster optimization)
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=5,
                validation_split=0.2,
                verbose=0,
            )

            # Evaluate on test set
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            return accuracy

        except Exception as e:
            print(f"Error: {e}")
            return 0.0

    # Create optimizer
    print("\n[OPTIMIZER] Creating IDT-R Optimizer...")
    optimizer = IDTROptimizer(
        search_space=search_space,
        max_iterations=12,
        n_random_init=6,
        n_top_leaves=3,
        n_samples_per_leaf=2,
        tree_max_depth=4,
        verbose=True,
        maximize=True,
        seed=42,
    )

    # Run optimization
    print("\n[RUNNING] Running optimization (this may take a few minutes)...")
    best_params, best_score = optimizer.optimize(objective)

    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"[OK] Best Validation Accuracy: {best_score:.6f}")
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
