# Package Verification Guide

This document provides step-by-step instructions to verify that the IDT-R Optimizer package is correctly implemented and functional.

## Quick Verification (5 minutes)

### 1. Check Package Structure

```bash
cd idt_r_optimizer
ls -R
```

Should show:
```
idt_r/
  __init__.py
  optimizer.py
  search_space.py
  tree_utils.py
  sampler.py
  history.py
  utils.py
examples/
  example_svm.py
  example_rf.py
tests/
  test_basic.py
```

### 2. Verify Imports Work

```python
# Python shell
from idt_r_optimizer import IDTROptimizer, SearchSpace

print("✓ IDTROptimizer imported successfully")
print("✓ SearchSpace imported successfully")
```

### 3. Create Simple Optimizer

```python
from idt_r_optimizer import IDTROptimizer

space = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
optimizer = IDTROptimizer(space, max_iterations=3, n_random_init=2, verbose=False)

print("✓ IDTROptimizer instantiated successfully")
```

## Full Verification (15 minutes)

### 1. Run Tests

```bash
pip install pytest
pytest tests/test_basic.py -v
```

Expected output:
- All tests pass (green checkmarks)
- Coverage > 85%
- No warnings or errors

### 2. Run Example Scripts

#### SVM Example
```bash
python examples/example_svm.py
```

Expected output:
- Loads Iris dataset
- Runs IDT-R optimization
- Shows best parameters and score
- Summary statistics
- **Best Score should be > 0.9**

#### Random Forest Example
```bash
python examples/example_rf.py
```

Expected output:
- Generates synthetic dataset
- Runs IDT-R optimization
- Runs Random Search baseline
- Comparison showing IDT-R advantage
- **IDT-R should show improvement over baseline**

### 3. Manual Functionality Test

```python
import numpy as np
from idt_r_optimizer import IDTROptimizer

# Define quadratic function with optimum at (0.7, 0.3)
space = {
    "x": (0.0, 1.0),
    "y": (0.0, 1.0),
}

def objective(params):
    x = params["x"]
    y = params["y"]
    return 1.0 - ((x - 0.7)**2 + (y - 0.3)**2)

# Run optimization
optimizer = IDTROptimizer(
    space,
    max_iterations=10,
    n_random_init=3,
    verbose=False,
)

best_params, best_score = optimizer.optimize(objective)

# Verify results
assert best_score > 0.5, "Score should be > 0.5"
assert abs(best_params["x"] - 0.7) < 0.3, "Should find x near 0.7"
assert abs(best_params["y"] - 0.3) < 0.3, "Should find y near 0.3"

print(f"✓ Found optimal point with score {best_score:.4f}")
print(f"✓ Params: x={best_params['x']:.4f}, y={best_params['y']:.4f}")
```

## Installation Verification

### Install from Source

```bash
cd idt_r_optimizer
pip install -e .
```

Check: `pip show idt-r-optimizer`

Expected output:
```
Name: idt-r-optimizer
Version: 0.1.0
Summary: Advanced hyperparameter optimization using Iterative Decision Tree - Random (IDT-R) algorithm
Location: /path/to/idt_r_optimizer
```

### Verify Dependencies

```python
import numpy
import sklearn
print(f"NumPy version: {numpy.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
```

Both should be recent versions (numpy >= 1.20.0, sklearn >= 1.0.0)

## API Verification

### Test Each Module

#### 1. SearchSpace
```python
from idt_r_optimizer import SearchSpace

# Test creation
space = SearchSpace({
    "x": (0.0, 1.0),
    "n": (10, 100),
    "choice": ["A", "B", "C"],
})

# Test sampling
samples = space.sample_uniform(5)
assert len(samples) == 5

# Test normalization
params = {"x": 0.5, "n": 55, "choice": "B"}
normalized = space.normalize(params)
denormalized = space.denormalize(normalized)

assert abs(denormalized["x"] - 0.5) < 1e-5
assert denormalized["n"] == 55
assert denormalized["choice"] == "B"

print("✓ SearchSpace works correctly")
```

#### 2. OptimizationHistory
```python
from idt_r_optimizer import OptimizationHistory

history = OptimizationHistory(maximize=True)

history.add_record(1, {"x": 0.5}, 0.8)
history.add_record(2, {"x": 0.7}, 0.9)
history.add_record(3, {"x": 0.3}, 0.7)

assert history.size() == 3
assert history.get_best_score() == 0.9
assert history.get_best_params()["x"] == 0.7

summary = history.get_summary()
assert summary["n_evaluations"] == 3

print("✓ OptimizationHistory works correctly")
```

#### 3. TreeNodeExtractor
```python
from idt_r_optimizer import TreeNodeExtractor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Train simple tree
X = np.array([[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.8, 0.9]], dtype=np.float32)
y = np.array([0.5, 0.7, 0.8, 0.9], dtype=np.float32)

tree = DecisionTreeRegressor(max_depth=2)
tree.fit(X, y)

# Extract leaves
leaves = TreeNodeExtractor.extract_leaf_nodes(tree, ["param_0", "param_1"])

assert len(leaves) > 0
assert all(hasattr(leaf, 'param_bounds') for leaf in leaves)

# Rank leaves
ranked = TreeNodeExtractor.rank_leaves_by_prediction(leaves, maximize=True)
assert ranked[0].predicted_value >= ranked[-1].predicted_value

print("✓ TreeNodeExtractor works correctly")
```

#### 4. LeafSampler
```python
from idt_r_optimizer import LeafSampler, SearchSpace

space = SearchSpace({
    "x": (0.0, 1.0),
    "y": (0.0, 1.0),
})

leaf_bounds = {
    "x": (0.3, 0.7),
    "y": (0.4, 0.8),
}

samples = LeafSampler.sample_from_leaf_bounds(
    space, leaf_bounds, n_samples=10
)

assert len(samples) == 10
for sample in samples:
    assert 0.3 <= sample["x"] <= 0.7
    assert 0.4 <= sample["y"] <= 0.8

print("✓ LeafSampler works correctly")
```

## Performance Verification

### Test Convergence

```python
import numpy as np
from idt_r_optimizer import IDTROptimizer

# Rastrigin function (many local minima)
def rastrigin(params):
    x, y = params["x"], params["y"]
    return -(20 + x**2 - 10*np.cos(2*np.pi*x) + 
             y**2 - 10*np.cos(2*np.pi*y))

optimizer = IDTROptimizer(
    {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
    max_iterations=20,
    n_random_init=5,
    verbose=False,
)

best_params, best_score = optimizer.optimize(rastrigin)

history = optimizer.get_history()
scores = history.get_all_scores()

# Check convergence
initial_mean = np.mean(scores[:5])
final_mean = np.mean(scores[-5:])

print(f"Initial mean score: {initial_mean:.4f}")
print(f"Final mean score: {final_mean:.4f}")
print(f"Best score: {best_score:.4f}")

assert final_mean >= initial_mean * 0.8, "Should show improvement"
print("✓ Optimization shows convergence")
```

## Documentation Verification

Verify that all documentation files are present and readable:

```bash
ls -la
# Should show:
# README.md (comprehensive)
# QUICKSTART.md (quick start)
# ARCHITECTURE.md (design)
# CONTRIBUTING.md (contribution)
# CHANGELOG.md (history)
# IMPLEMENTATION_SUMMARY.md (summary)
```

Check each file opens and contains expected content:
- README.md: Has algorithm explanation, examples, API reference
- QUICKSTART.md: Has 5-minute quick start
- ARCHITECTURE.md: Has module descriptions and design
- CONTRIBUTING.md: Has contribution guidelines

## Code Quality Checks

### Type Hints
```bash
python -m mypy idt_r/optimizer.py --ignore-missing-imports
```

Should have minimal errors (if any).

### Code Style
```bash
python -m black --check idt_r/
python -m flake8 idt_r/
```

Should pass with no major issues.

## Reproducibility Check

```python
from idt_r_optimizer import IDTROptimizer

def objective(params):
    return params["x"]**2 + params["y"]**2

# Run 1
optimizer1 = IDTROptimizer(
    {"x": (0.0, 1.0), "y": (0.0, 1.0)},
    max_iterations=5,
    n_random_init=3,
    seed=42,
    verbose=False,
)
best1, score1 = optimizer1.optimize(objective)

# Run 2
optimizer2 = IDTROptimizer(
    {"x": (0.0, 1.0), "y": (0.0, 1.0)},
    max_iterations=5,
    n_random_init=3,
    seed=42,
    verbose=False,
)
best2, score2 = optimizer2.optimize(objective)

# Should be identical
assert abs(score1 - score2) < 1e-10
print("✓ Reproducibility with seed verified")
```

## Final Checklist

- [ ] Package structure correct
- [ ] All imports work
- [ ] Tests pass
- [ ] Examples run successfully
- [ ] API functions as documented
- [ ] Convergence observed
- [ ] Reproducibility works
- [ ] Documentation complete
- [ ] All files present
- [ ] Code quality acceptable

## Troubleshooting

### Import Error: "No module named 'idt_r'"
**Solution**: Install package in editable mode: `pip install -e .`

### Test Failures
**Solution**: Check test output, look for specific assertion errors

### Example Script Hangs
**Solution**: Check objective function for infinite loops or deadlocks

### Poor Optimization Performance
**Solution**: 
- Increase `n_top_leaves`
- Increase `n_samples_per_leaf`
- Increase `max_iterations`
- Check objective function for correctness

---

**All verifications passed? ✅ You're good to go!**

The IDT-R Optimizer package is fully functional and ready to use.
