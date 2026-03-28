# IDT-R Optimizer Examples

This directory contains three practical examples demonstrating the IDT-R optimizer on real-world machine learning and deep learning tasks.

## Example 1: SVM Hyperparameter Tuning

**File**: `example_svm.py`

Optimizes Support Vector Machine (SVM) hyperparameters on the scikit-learn digits dataset:

- **Dataset**: Digits (1,797 samples, 64 features, 10 classes)
- **Hyperparameters**:
  - `C`: Regularization parameter (continuous: 0.1-100.0)
  - `kernel`: Kernel type (categorical: linear, poly, rbf)
  - `gamma`: Kernel coefficient (continuous: 0.0001-1.0)
  - `degree`: Polynomial degree (discrete: 2-5)
- **Metric**: Cross-validation accuracy (5-fold CV)

**Run**:
```bash
python examples/example_svm.py
```

## Example 2: Random Forest Hyperparameter Tuning

**File**: `example_rf.py`

Optimizes Random Forest classifier hyperparameters on the scikit-learn digits dataset:

- **Dataset**: Digits (1,797 samples, 64 features, 10 classes)
- **Hyperparameters**:
  - `n_estimators`: Number of trees (continuous: 10-200)
  - `max_depth`: Maximum tree depth (continuous: 3-30)
  - `min_samples_split`: Minimum samples to split (continuous: 2-15)
  - `min_samples_leaf`: Minimum samples in leaf (continuous: 1-8)
  - `max_features`: Features per split (categorical: sqrt, log2)
  - `criterion`: Split criterion (categorical: gini, entropy)
- **Metric**: Cross-validation accuracy (3-fold CV)

**Run**:
```bash
python examples/example_rf.py
```

## Example 3: Neural Network Hyperparameter Tuning

**File**: `example_neural_network.py`

Optimizes deep learning hyperparameters for a neural network on the MNIST dataset:

- **Dataset**: MNIST (10,000 training samples, 2,000 test samples, 10 classes)
- **Architecture**: 
  - Input: 784 features (28×28 pixels flattened)
  - Hidden layer 1: Variable units with dropout
  - Hidden layer 2: 64 units with dropout
  - Output: 10 units (softmax)
- **Hyperparameters**:
  - `learning_rate`: Learning rate (continuous: 0.0001-0.01)
  - `batch_size`: Batch size (discrete: 16-128)
  - `hidden_units`: First hidden layer units (continuous: 64-512)
  - `dropout_rate`: Dropout rate (continuous: 0.1-0.5)
  - `optimizer`: Optimizer type (categorical: adam, sgd, rmsprop)
- **Metric**: Test accuracy
- **Note**: Requires TensorFlow: `pip install tensorflow`

**Run**:
```bash
python examples/example_neural_network.py
```

## Key Features Demonstrated

### 1. Multiple Data Types
- **Continuous**: C, gamma, learning_rate, dropout_rate, etc.
- **Discrete**: batch_size, hidden_units, n_estimators, max_depth, etc.
- **Categorical**: kernel, optimizer, criterion, etc.

### 2. Different ML Paradigms
- **Shallow ML**: SVM and Random Forest (traditional)
- **Deep Learning**: Neural Networks (modern)

### 3. Cross-Validation
- SVM: 5-fold CV
- Random Forest: 3-fold CV
- Neural Network: Internal validation split

### 4. Performance Metrics
- Scikit-learn models: Accuracy scores
- Deep learning: Validation and test accuracy

## Running All Examples

```bash
cd idt_r_optimizer/examples

# Run SVM example
python example_svm.py

# Run Random Forest example
python example_rf.py

# Run Neural Network example (requires TensorFlow)
python example_neural_network.py
```

## Comparing Results

Each example outputs:
- **Search Space**: Parameter names and ranges
- **Optimization Progress**: Iteration-by-iteration results
- **Best Results**: 
  - Best score achieved
  - Best hyperparameters
  - Total evaluations and statistics
  - Mean and standard deviation of scores across all evaluations

## Notes

- All examples use fixed random seeds (seed=42) for reproducibility
- Examples use relatively small iteration counts (12-20) for demonstration; increase for better results
- The neural network example may take several minutes due to MNIST download and model training
- For faster experimentation, consider reducing dataset sizes or tree depth parameters
