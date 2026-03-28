# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-03-28

### Fixed
- **Corrected Journal Citation**: Updated to proper foundational paper attribution
  - Paper: "Hyperparameter Optimization Using Iterative Decision Tree (IDT)"
  - Authors: Narith Saum, Satoshi Sugiura, Mongkut Piantanakulchai
  - Journal: IEEE Access, Vol. 10, Pages 106812-106827, 2022
  - DOI: 10.1109/ACCESS.2022.3212387

### Enhanced
- **CITATIONS.md**: Updated with correct author names and paper title
- **README.md**: Corrected citation BibTeX with full author names
- **ARCHITECTURE.md**: Added proper attribution to foundational IDT paper
- **Academic Attribution**: Proper credit to original algorithm authors

## [0.1.1] - 2026-03-28

### Added
- **Comprehensive Citation Documentation**: 
  - New `CITATIONS.md` file with multiple citation formats (BibTeX, APA, Chicago)
  - IEEE Access journal paper reference
  - Core theory references (Breiman, Quinlan, Bergstra & Bengio, Hutter et al.)
  - Implementation library acknowledgments and licenses

### Enhanced
- **README.md**: Expanded Citation section with software and research paper citations
- **ARCHITECTURE.md**: Improved References section with proper academic citations and links
- **Documentation**: Better guidance on citing the package in academic work

## [0.1.0] - 2024-01-01

### Added
- **Core IDT-R Algorithm**: Complete implementation of Iterative Decision Tree - Random optimization
  - Decision tree surrogate model training
  - Intelligent leaf node extraction with interval bounds
  - Constrained random sampling within promising regions
  - Iterative refinement loop

- **Search Space Support**:
  - Continuous (float) parameters
  - Discrete (integer) parameters
  - Categorical (choice) parameters
  - Mixed parameter types in single optimization

- **Modules**:
  - `optimizer.py`: Main IDTROptimizer class
  - `search_space.py`: Parameter space definition and normalization
  - `tree_utils.py`: Decision tree node extraction logic
  - `sampler.py`: Random sampling within leaf bounds
  - `history.py`: Evaluation tracking and result management
  - `utils.py`: Helper functions and utilities

- **Documentation**:
  - Comprehensive README with algorithm explanation
  - API reference documentation
  - Quick start guide
  - Usage examples

- **Examples**:
  - SVM hyperparameter tuning example
  - Random Forest tuning with baseline comparison
  - Dataset and cross-validation integration

- **Tests**:
  - Unit tests for all modules
  - Integration tests
  - Parameter type coverage
  - Reproducibility tests

- **PyPI Compatibility**:
  - setup.py for pip installation
  - pyproject.toml with modern Python package configuration
  - requirements.txt with dependencies
  - MIT License

### Key Features
- ✅ Production-ready code quality
- ✅ Comprehensive error handling
- ✅ Type hints for better IDE support
- ✅ Verbose logging for transparency
- ✅ Reproducible with seed support
- ✅ Efficient vectorized operations
- ✅ Memory-conscious design

### Dependencies
- numpy >= 1.20.0
- scikit-learn >= 1.0.0

---

## Future Roadmap

### v0.2.0 (Planned)
- [ ] Warm-start from previous optimizations
- [ ] Parallel evaluation support
- [ ] Custom surrogate model option
- [ ] Constraint support
- [ ] Early stopping criteria

### v0.3.0 (Planned)
- [ ] Multi-objective optimization
- [ ] Ensemble surrogate models
- [ ] Advanced sampling strategies
- [ ] Performance profiling utilities

### v1.0.0 (Planned)
- Stable API
- Production validation
- Performance benchmarks
- Large-scale case studies
