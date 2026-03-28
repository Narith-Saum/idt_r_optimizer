# IDT-R Architecture Overview

## High-Level Design

The IDT-R optimizer is designed with clean, modular architecture following best practices for production-ready Python packages.

```
IDTROptimizer (Main Orchestrator)
├── SearchSpace (Parameter Definition)
├── OptimizationHistory (Result Tracking)
├── Surrogate Model (DecisionTreeRegressor)
├── TreeNodeExtractor (Tree Analysis)
├── LeafSampler (Candidate Generation)
└── Utils (Helper Functions)
```

## Core Modules

### 1. search_space.py
**Responsibility**: Define and normalize hyperparameter spaces

**Key Classes**:
- `SearchSpace`: Main interface for working with mixed parameter types
- `ContinuousVariable`: Float parameters with bounds
- `DiscreteVariable`: Integer parameters with bounds
- `CategoricalVariable`: Discrete choice parameters

**Key Operations**:
- Normalize: Convert actual values to [0, 1] space
- Denormalize: Convert [0, 1] back to actual values
- Sample: Generate random candidates

**Design Decision**: 
- Normalization allows uniform treatment of different parameter types
- All tree operations work in [0, 1] space internally
- Type-aware denormalization ensures correct final values

### 2. history.py
**Responsibility**: Track evaluation results and analysis

**Key Classes**:
- `OptimizationHistory`: Track all evaluations
- `EvaluationRecord`: Single evaluation datapoint

**Key Features**:
- Deduplication detection with tolerance
- Best result tracking (maximize or minimize)
- Summary statistics
- Full history access

**Design Decision**:
- Tolerance-based deduplication handles floating-point comparison
- Immutable records prevent accidental modifications
- Separate track of "evaluated params" for fast lookup

### 3. tree_utils.py
**Responsibility**: Extract information from trained decision tree

**Key Classes**:
- `TreeNodeExtractor`: Static methods for tree analysis
- `LeafNode`: Represents a leaf with bounds and predicted value

**Key Operations**:
- `extract_leaf_nodes()`: Get all leaves with their hyperparameter bounds
- `rank_leaves_by_prediction()`: Sort leaves by predicted performance
- `get_top_leaves()`: Select best N leaves

**Design Decision**:
- Direct tree traversal avoids sklearn internals
- Interval extraction is deterministic and reproducible
- Leaves ranked by average predicted value (greedy strategy)

### 4. sampler.py
**Responsibility**: Generate candidate points from leaf regions

**Key Classes**:
- `LeafSampler`: Static methods for sampling

**Key Operations**:
- `sample_from_leaf_bounds()`: Random uniform sampling within leaf
- `sample_from_multiple_leaves()`: Sample from multiple regions
- `sample_uniform_global()`: Baseline uniform sampling

**Design Decision**:
- Uniform sampling within bounds (unbiased)
- Respects variable types through SearchSpace.denormalize()
- Can sample from multiple leaves simultaneously

### 5. optimizer.py
**Responsibility**: Main optimization loop orchestration

**Key Classes**:
- `IDTROptimizer`: Main interface

**Key Methods**:
- `optimize()`: Run full optimization
- `_train_surrogate()`: Train decision tree
- `_extract_leaves()`: Get leaf nodes
- `_select_top_leaves()`: Choose best regions
- `_sample_from_leaves()`: Generate candidates
- `_evaluate_candidates()`: Evaluate and track

**Design Decision**:
- Clear separation of concerns with private methods
- Extensible architecture for future improvements
- Verbose logging for transparency
- Flexible API with sensible defaults

### 6. utils.py
**Responsibility**: Utility functions and helpers

**Key Functions**:
- `remove_duplicate_params()`: Deduplication
- `filter_new_params()`: Find unevaluated candidates
- `normalize_scores()`: Score normalization
- `format_params_for_display()`: Logging helper

**Design Decision**:
- Small, focused utility functions
- No global state or side effects
- Ready for extension without breaking changes

## Data Flow

### Initialization
```
User Params
    ↓
SearchSpace Validation
    ↓
Initialize OptimizationHistory
    ↓
Random Initial Sampling
    ↓
Evaluate & Record
```

### Iteration Loop
```
Get All Evaluated Data (X, y)
    ↓
Train DecisionTreeRegressor
    ↓
Extract Leaf Nodes & Bounds
    ↓
Rank Leaves by Prediction
    ↓
Select Top N Leaves
    ↓
Sample from Leaf Bounds
    ↓
Remove Duplicates
    ↓
Filter Already-Evaluated
    ↓
Evaluate New Candidates
    ↓
Update History
```

## Key Design Principles

### 1. Modularity
- Each module has a single responsibility
- Modules are loosely coupled
- Easy to test and extend

### 2. Type Safety
- Type hints throughout codebase
- Helps IDE autocomplete and error detection
- Makes code more maintainable

### 3. Immutability
- Records are read-only dataclasses
- History tracking is append-only
- Prevents accidental state mutations

### 4. Reproducibility
- Seed parameter affects all randomness
- Same seed = same results
- Important for research and debugging

### 5. Clarity
- Verbose logging by default
- Descriptive variable names
- Clear method names

### 6. Performance
- Vectorized operations where possible
- Efficient duplicate detection
- Minimal object creation

## Extension Points

### Adding New Surrogate Models
Modify `_train_surrogate()` in `optimizer.py`:
```python
def _train_surrogate(self):
    # Replace DecisionTreeRegressor with another model
    self.model = MyCustomSurrogate()
    self.model.fit(X, y)
```

### Custom Sampling Strategies
Add to `sampler.py`:
```python
class AdvancedSampler:
    @staticmethod
    def sample_with_importance():
        # Weight sampling by leaf importance
        pass
```

### Constraint Support
Add to `search_space.py`:
```python
class ConstrainedSearchSpace(SearchSpace):
    def __init__(self, space_dict, constraints):
        super().__init__(space_dict)
        self.constraints = constraints
```

## Performance Characteristics

### Time Complexity per Iteration
- Tree training: O(n_evaluations × log(n_params))
- Leaf extraction: O(tree_depth)
- Sampling: O(n_top_leaves × n_samples_per_leaf)
- Overall: Dominated by tree training

### Space Complexity
- History: O(n_evaluations × n_params)
- Tree: O(n_leaves)
- Typically: 100-1000 evaluations stored

### Typical Evaluation Counts
- Initial random: ~5-20
- Per iteration: ~5-20
- Total (20 iterations): ~100-400 evaluations

## Testing Strategy

### Unit Tests (test_basic.py)
- Test each module independently
- Cover normal cases and edge cases
- Test type conversions and normalizations

### Integration Tests
- Test full optimization pipeline
- Test with different parameter types
- Verify reproducibility

### Property Tests (Future)
- Test invariants that should always hold
- Use hypothesis for property-based testing

## Documentation

### For Users
- README.md: Overview and examples
- QUICKSTART.md: Get started in 5 minutes
- API docstrings: Detailed function documentation

### For Developers
- This file (ARCHITECTURE.md): Design overview
- CONTRIBUTING.md: Contribution guidelines
- Code comments: Implementation details

## Future Improvements

1. **Parallel Evaluation**: Evaluate multiple candidates in parallel
2. **Warm Start**: Initialize from previous optimizations
3. **Custom Surrogates**: User-defined surrogate models
4. **Constraints**: Add constraint support
5. **Multi-Objective**: Pareto frontier optimization
6. **Early Stopping**: Stop when improvement plateaus

## Dependencies

### Core
- **numpy**: Numerical operations
- **scikit-learn**: DecisionTreeRegressor surrogate

### Development
- **pytest**: Unit testing
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

## References

### Core Algorithm - IDT (Iterative Decision Tree)
- **Saum, N., Sugiura, S., & Piantanakulchai, M.** (2022). "Hyperparameter Optimization Using Iterative Decision Tree (IDT)." *IEEE Access*, 10, 106812-106827. https://doi.org/10.1109/ACCESS.2022.3212387
  - Foundational work on the Iterative Decision Tree algorithm for hyperparameter optimization

### Core ML Algorithms
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
- Quinlan, J. R. (1986). "Induction of Decision Trees." *Machine Learning*, 1(1), 81-106.

### Hyperparameter Optimization Context
- Bergstra, J., & Bengio, Y. (2012). "Random search for hyper-parameter optimization." *Journal of Machine Learning Research*, 13(1), 281-305.
- Hutter, F., Lücke, J., & Schmidt-Thieme, L. (2015). "Beyond manual tuning of hyperparameters." *KI – Künstliche Intelligenz*, 29(4), 329-337.

### Implementation
- [Scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [NumPy Documentation](https://numpy.org/)
- [Python Packaging Guide](https://packaging.python.org/)

---

**Last Updated**: 2024-01-01
**Maintainer**: IDT-R Development Team
