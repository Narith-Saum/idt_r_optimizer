"""
Test suite for idt_r_optimizer.

Run with: pytest tests/test_basic.py -v
"""

import pytest
import numpy as np
from idt_r_optimizer import (
    IDTROptimizer,
    SearchSpace,
    OptimizationHistory,
    ContinuousVariable,
    DiscreteVariable,
    CategoricalVariable,
)


class TestSearchSpace:
    """Test SearchSpace module."""

    def test_continuous_variable(self):
        var = ContinuousVariable("learning_rate", 0.001, 0.1)
        assert var.is_continuous()
        assert not var.is_discrete()
        assert not var.is_categorical()

        # Test normalization
        normalized = var.normalize(0.0505)
        assert 0 <= normalized <= 1
        denormalized = var.denormalize(normalized)
        assert abs(denormalized - 0.0505) < 1e-5

    def test_discrete_variable(self):
        var = DiscreteVariable("n_estimators", 10, 100)
        assert not var.is_continuous()
        assert var.is_discrete()
        assert not var.is_categorical()

        # Test normalization
        normalized = var.normalize(55)
        assert 0 <= normalized <= 1
        denormalized = var.denormalize(normalized)
        assert isinstance(denormalized, int)
        assert 10 <= denormalized <= 100

    def test_categorical_variable(self):
        var = CategoricalVariable("kernel", ["linear", "rbf", "poly"])
        assert not var.is_continuous()
        assert not var.is_discrete()
        assert var.is_categorical()

        # Test normalization
        normalized = var.normalize("rbf")
        assert 0 <= normalized <= 1
        denormalized = var.denormalize(normalized)
        assert denormalized == "rbf"

    def test_search_space_basic(self):
        space_dict = {
            "learning_rate": (0.001, 0.1),
            "n_estimators": (10, 500),
            "kernel": ["linear", "rbf"],
        }
        space = SearchSpace(space_dict)

        assert len(space.get_param_names()) == 3
        assert space.get_bounds() == [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    def test_search_space_sampling(self):
        space = SearchSpace({
            "x": (0.0, 1.0),
            "y": (0.0, 10.0),
        })

        samples = space.sample_uniform(10)
        assert len(samples) == 10
        assert all(isinstance(s, dict) for s in samples)
        assert all("x" in s and "y" in s for s in samples)

    def test_search_space_normalize_denormalize(self):
        space = SearchSpace({
            "a": (1.0, 5.0),
            "b": (10, 20),
            "c": ["A", "B", "C"],
        })

        params = {"a": 3.0, "b": 15, "c": "B"}
        normalized = space.normalize(params)
        denormalized = space.denormalize(normalized)

        assert abs(denormalized["a"] - 3.0) < 1e-5
        assert denormalized["b"] == 15
        assert denormalized["c"] == "B"


class TestOptimizationHistory:
    """Test OptimizationHistory module."""

    def test_history_basic(self):
        history = OptimizationHistory(maximize=True)

        assert history.size() == 0
        assert history.get_best_score() is None

        # Add records
        history.add_record(1, {"x": 0.5}, 0.8)
        history.add_record(2, {"x": 0.7}, 0.9)
        history.add_record(3, {"x": 0.3}, 0.7)

        assert history.size() == 3
        assert history.get_best_score() == 0.9

    def test_history_maximize_vs_minimize(self):
        # Maximize
        history_max = OptimizationHistory(maximize=True)
        history_max.add_record(1, {"x": 1}, 0.5)
        history_max.add_record(2, {"x": 2}, 0.9)
        assert history_max.get_best_score() == 0.9

        # Minimize
        history_min = OptimizationHistory(maximize=False)
        history_min.add_record(1, {"x": 1}, 0.5)
        history_min.add_record(2, {"x": 2}, 0.9)
        assert history_min.get_best_score() == 0.5

    def test_history_deduplication(self):
        history = OptimizationHistory()

        params1 = {"x": 0.5000001, "y": 1.0}
        params2 = {"x": 0.5, "y": 1.0}

        history.mark_evaluated(params1)
        # Should detect as duplicate (with tolerance)
        assert history.has_params(params2)

    def test_history_summary(self):
        history = OptimizationHistory()
        history.add_record(1, {"x": 1}, 0.5)
        history.add_record(2, {"x": 2}, 0.7)
        history.add_record(3, {"x": 3}, 0.6)

        summary = history.get_summary()
        assert summary["n_evaluations"] == 3
        assert summary["best_score"] == 0.7
        assert abs(summary["mean_score"] - 0.6) < 1e-5


class TestIDTROptimizer:
    """Test IDTROptimizer main class."""

    def test_optimizer_initialization(self):
        space = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        optimizer = IDTROptimizer(space, max_iterations=5)

        assert optimizer.max_iterations == 5
        assert optimizer.n_random_init == 5
        assert not optimizer.maximize is False

    def test_optimizer_simple_optimization(self):
        """Simple optimization: maximize -(x-0.7)^2 - (y-0.3)^2."""

        space = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
        }

        def objective(params):
            x = params["x"]
            y = params["y"]
            # Global maximum at (0.7, 0.3) with value ~1.0
            return 1.0 - ((x - 0.7) ** 2 + (y - 0.3) ** 2)

        optimizer = IDTROptimizer(
            space,
            max_iterations=10,
            n_random_init=5,
            verbose=False,
            maximize=True,
            seed=42,
        )

        best_params, best_score = optimizer.optimize(objective)

        # Should find score reasonably close to max (not exact due to sampling)
        assert best_score > 0.5
        assert best_params is not None
        assert "x" in best_params
        assert "y" in best_params

    def test_optimizer_history_tracking(self):
        space = {"x": (0.0, 1.0)}

        def objective(params):
            return params["x"]

        optimizer = IDTROptimizer(
            space,
            max_iterations=3,
            n_random_init=2,
            verbose=False,
        )

        optimizer.optimize(objective)

        history = optimizer.get_history()
        assert history.size() > 0
        assert len(history.get_all_scores()) == history.size()
        assert len(history.get_all_params()) == history.size()

    def test_optimizer_with_categorical(self):
        space = {
            "x": (0.0, 1.0),
            "choice": ["A", "B", "C"],
        }

        def objective(params):
            base_score = params["x"]
            if params["choice"] == "A":
                return base_score + 0.1
            elif params["choice"] == "B":
                return base_score + 0.2
            else:
                return base_score

        optimizer = IDTROptimizer(
            space,
            max_iterations=5,
            n_random_init=3,
            verbose=False,
        )

        best_params, best_score = optimizer.optimize(objective)
        assert best_score > 0
        assert best_params["choice"] in ["A", "B", "C"]

    def test_optimizer_with_mixed_types(self):
        space = {
            "learning_rate": (0.001, 0.1),  # Continuous
            "batch_size": (8, 256),          # Discrete
            "optimizer": ["adam", "sgd"],    # Categorical
        }

        call_count = 0

        def objective(params):
            nonlocal call_count
            call_count += 1
            return np.random.random()

        optimizer = IDTROptimizer(
            space,
            max_iterations=3,
            n_random_init=2,
            n_top_leaves=1,
            n_samples_per_leaf=1,
            verbose=False,
        )

        best_params, best_score = optimizer.optimize(objective)

        assert call_count > 0
        assert "learning_rate" in best_params
        assert "batch_size" in best_params
        assert "optimizer" in best_params

    def test_optimizer_get_best(self):
        space = {"x": (0.0, 1.0)}

        def objective(params):
            return params["x"]

        optimizer = IDTROptimizer(
            space,
            max_iterations=2,
            n_random_init=3,
            verbose=False,
        )

        optimizer.optimize(objective)

        best_params, best_score = optimizer.get_best()
        assert best_params is not None
        assert best_score is not None

    def test_optimizer_summary(self):
        space = {"x": (0.0, 1.0)}

        def objective(params):
            return params["x"]

        optimizer = IDTROptimizer(
            space,
            max_iterations=2,
            n_random_init=2,
            verbose=False,
        )

        optimizer.optimize(objective)

        summary = optimizer.get_summary()
        assert summary["algorithm"] == "IDT-R"
        assert summary["n_evaluations"] > 0
        assert "best_score" in summary


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_optimization(self):
        """Full optimization pipeline."""
        space = {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        }

        # Rastrigin function (has many local minima)
        def objective(params):
            x = params["x"]
            y = params["y"]
            return -(
                20 + x**2 - 10 * np.cos(2 * np.pi * x) +
                y**2 - 10 * np.cos(2 * np.pi * y)
            )

        optimizer = IDTROptimizer(
            space,
            max_iterations=15,
            n_random_init=5,
            n_top_leaves=2,
            n_samples_per_leaf=2,
            verbose=False,
            maximize=True,
            seed=123,
        )

        best_params, best_score = optimizer.optimize(objective)

        # Should improve from random
        history = optimizer.get_history()
        scores = history.get_all_scores()
        initial_mean = np.mean(scores[:5])
        final_mean = np.mean(scores[-5:])

        # IDT-R should show improvement
        assert final_mean >= initial_mean * 0.8  # Allow some variance

    def test_reproducibility_with_seed(self):
        """Results should be reproducible with same seed."""
        space = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

        def objective(params):
            return params["x"] + params["y"]

        optimizer1 = IDTROptimizer(
            space, max_iterations=3, n_random_init=2, verbose=False, seed=42
        )
        best1, score1 = optimizer1.optimize(objective)

        optimizer2 = IDTROptimizer(
            space, max_iterations=3, n_random_init=2, verbose=False, seed=42
        )
        best2, score2 = optimizer2.optimize(objective)

        # Should get same results (within floating point precision)
        assert abs(score1 - score2) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
