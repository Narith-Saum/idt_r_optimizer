"""Tests for parallel candidate evaluation."""

import threading
import time

import pytest

from idt_r_optimizer import IDTROptimizer


def process_safe_objective(params):
    """Module-level objective that can be evaluated in a worker process."""
    return float(params["x"])


class TestParallelEvaluation:
    """Test opt-in parallel execution behavior."""

    def test_rejects_invalid_parallel_settings(self):
        search_space = {"x": (0, 10)}

        for n_jobs in (0, -2, True):
            with pytest.raises(ValueError, match="n_jobs"):
                IDTROptimizer(search_space, n_jobs=n_jobs)

        with pytest.raises(ValueError, match="parallel_backend"):
            IDTROptimizer(search_space, parallel_backend="invalid")

    def test_threaded_evaluation_uses_multiple_workers_and_preserves_order(self):
        search_space = {"x": (0, 10)}
        candidates = [{"x": value} for value in range(4)]
        worker_barrier = threading.Barrier(2)
        thread_ids = set()
        lock = threading.Lock()

        def objective(params):
            with lock:
                thread_ids.add(threading.get_ident())
            worker_barrier.wait(timeout=2)
            time.sleep(0.01)
            return float(params["x"])

        optimizer = IDTROptimizer(
            search_space,
            n_jobs=2,
            parallel_backend="threading",
            verbose=False,
        )
        optimizer._evaluate_candidates(objective, candidates)

        assert len(thread_ids) == 2
        assert optimizer.get_history().get_all_params() == candidates
        assert optimizer.get_history().get_all_scores() == [0.0, 1.0, 2.0, 3.0]

    def test_parallel_evaluation_keeps_successful_results_after_errors(self):
        search_space = {"x": (0, 10)}
        candidates = [{"x": value} for value in range(4)]

        def objective(params):
            if params["x"] == 2:
                raise RuntimeError("expected evaluation failure")
            return float(params["x"])

        optimizer = IDTROptimizer(
            search_space,
            n_jobs=2,
            parallel_backend="threading",
            verbose=False,
        )
        optimizer._evaluate_candidates(objective, candidates)

        assert optimizer.get_history().get_all_params() == [
            {"x": 0},
            {"x": 1},
            {"x": 3},
        ]
        assert optimizer.get_history().get_all_scores() == [0.0, 1.0, 3.0]

    def test_optimization_raises_when_all_candidate_evaluations_fail(self):
        search_space = {"x": (0, 10)}

        def objective(params):
            raise RuntimeError(f"failure for {params['x']}")

        optimizer = IDTROptimizer(
            search_space,
            max_iterations=0,
            n_random_init=2,
            n_jobs=2,
            parallel_backend="threading",
            verbose=False,
        )

        with pytest.raises(RuntimeError, match="no candidate was evaluated successfully"):
            optimizer.optimize(objective)

    def test_process_backend_evaluates_picklable_objective(self):
        search_space = {"x": (0, 10)}
        candidates = [{"x": value} for value in range(3)]
        optimizer = IDTROptimizer(
            search_space,
            n_jobs=2,
            parallel_backend="loky",
            verbose=False,
        )

        optimizer._evaluate_candidates(process_safe_objective, candidates)

        assert optimizer.get_history().get_all_params() == candidates
        assert optimizer.get_history().get_all_scores() == [0.0, 1.0, 2.0]

    def test_threaded_optimization_is_reproducible_for_deterministic_objectives(self):
        search_space = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

        def objective(params):
            return params["x"] + params["y"]

        optimizer_one = IDTROptimizer(
            search_space,
            max_iterations=3,
            n_random_init=4,
            n_jobs=2,
            parallel_backend="threading",
            seed=42,
            verbose=False,
        )
        best_params_one, best_score_one = optimizer_one.optimize(objective)

        optimizer_two = IDTROptimizer(
            search_space,
            max_iterations=3,
            n_random_init=4,
            n_jobs=2,
            parallel_backend="threading",
            seed=42,
            verbose=False,
        )
        best_params_two, best_score_two = optimizer_two.optimize(objective)

        assert best_params_one == best_params_two
        assert best_score_one == best_score_two
        assert (
            optimizer_one.get_history().get_all_scores()
            == optimizer_two.get_history().get_all_scores()
        )
