---
description: "Use when modifying IDT-R Python code, optimizer behavior, parallel candidate evaluation, tests, examples, package metadata, or releases."
name: "IDT-R Python Package Guidelines"
applyTo: ["idt_r/**/*.py", "tests/**/*.py", "examples/**/*.py", "idt_r_optimizer.py"]
---

# IDT-R Python Package Guidelines

- Preserve the public `IDTROptimizer` API and keep serial execution as the default with `n_jobs=1`.
- Keep candidate generation, surrogate fitting, deduplication, and history mutation in the main optimizer process.
- Evaluate only independent candidate batches in workers, then record successful results in proposal order.
- Use `joblib` for optimizer-level parallelism and support only the documented `threading` and `loky` backends.
- Do not introduce nested parallelism in examples. Set inner model and cross-validation worker counts to one when optimizer-level parallelism is enabled.
- Add focused tests for serial compatibility, ordered history, worker exceptions, and each supported parallel backend when changing evaluation behavior.
- Keep runtime versions synchronized in `pyproject.toml`, `setup.py`, `idt_r/__init__.py`, and `idt_r_optimizer.py`.
- Before a release, run tests, build both artifacts, run `twine check`, and install the built wheel in an isolated environment.