---
name: idt-r-parallel-release
description: "Use when adding or reviewing IDT-R parallel computing, n_jobs behavior, joblib backends, package releases, GitHub workflows, or PyPI publishing readiness."
argument-hint: "Implement or validate an IDT-R parallel or release task"
user-invocable: true
---

# IDT-R Parallel and Release Workflow

## When to Use

- Add or change `n_jobs`, `parallel_backend`, or objective evaluation behavior
- Review concurrency safety, ordered histories, or worker error handling
- Prepare, validate, tag, or publish an IDT-R release

## Parallel Implementation

1. Keep `n_jobs=1` as the serial default.
2. Run only independent candidate evaluations in workers.
3. Collect results in proposal order and update `OptimizationHistory` in the parent process.
4. Preserve per-candidate error isolation.
5. Test `threading` and `loky` paths with deterministic objectives.
6. Document process serialization requirements and nested-parallelism guidance.

## Release Validation

1. Synchronize runtime and package versions.
2. Run `python -m pytest tests test_quick.py -q`.
3. Run `python -m build` and `python -m twine check dist/*`.
4. Install the built wheel into a clean environment and verify both public import paths.
5. Confirm `git diff --check`, a clean conflict-marker scan, and intended staged files.

## Publishing Rules

- Push only after explicit user approval.
- Prefer PyPI Trusted Publishing through `.github/workflows/publish.yml`.
- Never request or write a PyPI token in chat or source control.
- If trusted publishing is unavailable, the user must enter a token directly into the terminal for a manual `twine upload`.