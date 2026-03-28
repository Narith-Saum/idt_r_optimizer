# Quick Start Guide - IDT-R Optimizer

Get up and running with IDT-R in 5 minutes!

## 1. Installation

```bash
# Option A: From PyPI (when released)
pip install idt-r-optimizer

# Option B: From source
git clone https://github.com/yourname/idt_r_optimizer.git
cd idt_r_optimizer
pip install -e .
```

## 2. Basic Usage

### Minimal Example

```python
from idt_r_optimizer import IDTROptimizer

# Define your search space
search_space = {
    "learning_rate": (0.001, 0.1),
    "batch_size": (8, 256),
}

# Define your objective function
def objective(params):
    # Your training/evaluation code here
    score = train_model(**params)
    return score  # Return a single float

# Create optimizer
optimizer = IDTROptimizer(
    search_space=search_space,
    max_iterations=20,
    verbose=True,
)

# Run optimization
best_params, best_score = optimizer.optimize(objective)
print(f"Best Score: {best_score}")
print(f"Best Params: {best_params}")
```

## 3. Common Patterns

### Machine Learning Model Tuning

```python
from idt_r_optimizer import IDTROptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

search_space = {
    "n_estimators": (10, 200),
    "max_depth": (3, 30),
    "min_samples_leaf": (1, 10),
}

def objective(params):
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

optimizer = IDTROptimizer(search_space, max_iterations=15)
best_params, best_score = optimizer.optimize(objective)
```

### Neural Network Hyperparameter Tuning

```python
from idt_r_optimizer import IDTROptimizer

search_space = {
    "learning_rate": (0.0001, 0.01),
    "dropout": (0.0, 0.5),
    "hidden_units": (64, 512),
    "activation": ["relu", "tanh"],
}

def objective(params):
    model = create_model(**params)
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    return max(history.history['val_accuracy'])

optimizer = IDTROptimizer(search_space, max_iterations=20, maximize=True)
best_params, best_score = optimizer.optimize(objective)
```

## 4. Parameter Types

IDT-R supports three types of parameters:

```python
search_space = {
    # Continuous (float)
    "learning_rate": (0.001, 0.1),
    "gamma": (0.0, 1.0),
    
    # Discrete (integer - just use int bounds)
    "n_estimators": (10, 500),
    "max_depth": (3, 30),
    
    # Categorical (list of options)
    "optimizer": ["adam", "sgd", "rmsprop"],
    "kernel": ["linear", "rbf", "poly"],
}
```

## 5. Configuration

### For Fast/Cheap Objectives (seconds per evaluation)

```python
optimizer = IDTROptimizer(
    search_space,
    max_iterations=50,      # More iterations possible
    n_random_init=5,
    n_top_leaves=3,
    n_samples_per_leaf=3,
    tree_max_depth=5,
)
```

### For Slow/Expensive Objectives (hours per evaluation)

```python
optimizer = IDTROptimizer(
    search_space,
    max_iterations=5,       # Fewer iterations to save time
    n_random_init=20,       # Better initial sampling
    n_top_leaves=2,         # Focus on best regions
    n_samples_per_leaf=1,   # Fewer per leaf
    tree_max_depth=3,       # Simpler tree
)
```

## 6. Advanced Usage

### Access Optimization History

```python
best_params, best_score = optimizer.optimize(objective)

history = optimizer.get_history()

# Get all evaluated parameters and scores
all_params = history.get_all_params()
all_scores = history.get_all_scores()

# Get summary statistics
summary = history.get_summary()
print(f"Mean Score: {summary['mean_score']}")
print(f"Best Score: {summary['best_score']}")

# Find best record
best_record = history.get_best()
print(f"Best found at iteration: {best_record.iteration}")
```

### Minimize Instead of Maximize

```python
optimizer = IDTROptimizer(
    search_space,
    maximize=False,  # Minimize loss/error
)

def objective(params):
    model = MyModel(**params)
    loss = evaluate_loss(model)
    return loss  # Optimizer will minimize this
```

### Reproducible Results with Seed

```python
optimizer = IDTROptimizer(
    search_space,
    seed=42,  # Set seed for reproducibility
)
```

## 7. Troubleshooting

### "Error evaluating params: ..."

Your objective function raised an exception. Make sure:
- It accepts a dictionary with all parameters
- It returns a single float value
- It handles edge cases gracefully

### Low Performance

Try:
- Increase `n_random_init` for better initial understanding
- Reduce `tree_max_depth` to avoid overfitting
- Increase `n_samples_per_leaf` for better exploration
- Let it run for more iterations

### Slow Optimization

If your objective is very fast:
- You can safely use more iterations
- But the overhead isn't significant since evaluation dominates

## 8. Next Steps

- Read the [README](README.md) for detailed documentation
- Check [examples](examples/) for complete working code
- Review [API Reference](README.md#api-reference)
- Run tests: `pytest tests/ -v`

## 9. Common Questions

**Q: How many evaluations does IDT-R need?**
A: Typically 5x-10x fewer than random search, 2x-3x fewer than Bayesian optimization.

**Q: Can I use IDT-R for continuous and categorical mixed spaces?**
A: Yes! That's a core feature.

**Q: Is IDT-R deterministic?**
A: No, but it's reproducible with `seed` parameter.

**Q: How do I choose hyperparameters?**
A: Start with defaults and adjust based on problem:
- More iterations if you have time budget
- Fewer iterations if each evaluation is very expensive
- Increase samples per leaf for better exploration

**Q: Can I run IDT-R in parallel?**
A: Not yet, but it's planned for v0.2.0.

---

**Need more help?** Check the [examples](examples/) or [open an issue](https://github.com/yourname/idt_r_optimizer/issues)!
