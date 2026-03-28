# Citations and References

## Citing IDT-R Optimizer

If you use the IDT-R Optimizer package in your research or applications, please cite it as follows:

### BibTeX

```bibtex
@software{idt_r_2024,
  title={IDT-R Optimizer: Iterative Decision Tree - Random Hyperparameter Optimization},
  author={{IDT-R Development Team}},
  year={2024},
  url={https://github.com/yourname/idt_r_optimizer},
  version={0.1.0},
  howpublished={Python Package Index (PyPI)}
}
```

### APA Style

IDT-R Development Team. (2024). *IDT-R Optimizer: Iterative Decision Tree - Random Hyperparameter Optimization* (Version 0.1.0) [Software]. Retrieved from https://github.com/yourname/idt_r_optimizer

### Chicago Style

IDT-R Development Team. 2024. "IDT-R Optimizer: Iterative Decision Tree - Random Hyperparameter Optimization." Version 0.1.0. Accessed at https://github.com/yourname/idt_r_optimizer.

---

## Related Research Papers

### Foundational Paper - IDT (Iterative Decision Tree)

The IDT-R Optimizer is based on the Iterative Decision Tree algorithm published in IEEE Access:

**[Hyperparameter Optimization Using Iterative Decision Tree (IDT)](https://doi.org/10.1109/ACCESS.2022.3212387)**

- **Authors**: Narith Saum, Satoshi Sugiura, Mongkut Piantanakulchai
- **Journal**: IEEE Access
- **Volume**: 10
- **Pages**: 106812–106827
- **Year**: 2022
- **DOI**: 10.1109/ACCESS.2022.3212387
- **Publisher**: IEEE

This paper introduces the Iterative Decision Tree algorithm that forms the theoretical foundation for the IDT-R optimizer. It demonstrates how decision tree surrogates can be used to identify promising hyperparameter regions through intelligent leaf node extraction and constrained random sampling.

#### BibTeX Citation

```bibtex
@article{saum2022hyperparameter,
  title={Hyperparameter Optimization Using Iterative Decision Tree (IDT)},
  author={Saum, Narith and Sugiura, Satoshi and Piantanakulchai, Mongkut},
  journal={IEEE Access},
  volume={10},
  pages={106812--106827},
  year={2022},
  doi={10.1109/ACCESS.2022.3212387},
  publisher={IEEE}
}
```

---

## Core Theory References

### Decision Trees and Random Forests

1. **Breiman, L.** (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
   - Foundational work on random forests and decision trees

2. **Quinlan, J. R.** (1986). "Induction of Decision Trees." *Machine Learning*, 1(1), 81-106.
   - Seminal paper on decision tree learning algorithms

### Hyperparameter Optimization

3. **Bergstra, J., & Bengio, Y.** (2012). "Random search for hyper-parameter optimization." *Journal of Machine Learning Research*, 13(1), 281-305.
   - Theoretical foundation for random search in hyperparameter optimization

4. **Hutter, F., Lücke, J., & Schmidt-Thieme, L.** (2015). "Beyond manual tuning of hyperparameters." *KI – Künstliche Intelligenz*, 29(4), 329-337.
   - Overview of automated hyperparameter tuning methods

### Surrogate-Based Optimization

5. **Jones, D. R.** (2001). "A taxonomy of global optimization methods based on response surfaces." *Journal of Global Optimization*, 21(4), 345-383.
   - Theory of surrogate models and response surfaces

---

## Implementation Libraries

### Primary Dependencies

- **NumPy** (≥1.20.0): Fundamental package for numerical computing
  - [https://numpy.org/](https://numpy.org/)
  - License: BSD

- **Scikit-learn** (≥1.0.0): Machine learning library providing DecisionTreeRegressor
  - [https://scikit-learn.org/](https://scikit-learn.org/)
  - License: BSD

### Optional Dependencies

- **TensorFlow/Keras** (for deep learning examples): Neural network framework
  - [https://www.tensorflow.org/](https://www.tensorflow.org/)
  - License: Apache 2.0

---

## Academic Context

The IDT-R algorithm belongs to the class of **surrogate-based optimization methods**, which:

1. **Train a surrogate model** on evaluated points (using decision trees)
2. **Identify promising regions** from the surrogate structure
3. **Sample efficiently** within high-potential areas
4. **Iteratively refine** the search space

This approach balances:
- **Exploration**: Random sampling discovers new regions
- **Exploitation**: Tree structure guides toward promising regions
- **Computational efficiency**: Minimal surrogate modeling overhead

---

## Citation in Different Contexts

### For Academic Papers

Use the expanded citation with DOI reference:

> This work uses IDT-R (Optimal Decision Tree - Random) for hyperparameter optimization [cite software]. The optimization framework draws on techniques surveyed in [2022 IEEE Access paper on hyperparameter optimization](https://doi.org/10.1109/ACCESS.2022.3212387).

### For Technical Reports

Include version and access date:

> IDT-R Optimizer [0.1.0] was used for hyperparameter tuning. Source code: https://github.com/yourname/idt_r_optimizer (Accessed: YYYY-MM-DD).

### For Repositories

Add to README or documentation:

```markdown
## Acknowledgments

This project uses the IDT-R Optimizer package for hyperparameter tuning. The implementation is based on hyperparameter optimization techniques surveyed in [IEEE Access 2022](https://doi.org/10.1109/ACCESS.2022.3212387).
```

---

## Questions?

For questions about citations or references, please:
- Open an issue on GitHub
- Check the main [README.md](README.md)
- Review the [ARCHITECTURE.md](ARCHITECTURE.md)

---

*Last Updated: March 28, 2024*
