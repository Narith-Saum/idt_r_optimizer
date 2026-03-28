"""
History tracking for optimization results.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationRecord:
    """Records a single objective function evaluation."""

    iteration: int
    params: Dict[str, Any]
    score: float
    
    def __repr__(self) -> str:
        return f"EvaluationRecord(iter={self.iteration}, score={self.score:.6f})"


class OptimizationHistory:
    """Tracks evaluation history and provides utilities for deduplication."""

    def __init__(self, maximize: bool = True):
        """
        Initialize history tracker.
        
        Parameters:
        -----------
        maximize : bool
            Whether we are maximizing (True) or minimizing (False) the objective
        """
        self.records: List[EvaluationRecord] = []
        self.maximize = maximize
        self._best_record: Optional[EvaluationRecord] = None
        self._params_set: set = set()  # For quick duplicate checking

    def add_record(
        self,
        iteration: int,
        params: Dict[str, Any],
        score: float,
    ) -> None:
        """Add an evaluation record to history."""
        record = EvaluationRecord(iteration=iteration, params=params, score=score)
        self.records.append(record)

        # Update best record
        if self._best_record is None:
            self._best_record = record
        else:
            if self.maximize:
                if score > self._best_record.score:
                    self._best_record = record
            else:
                if score < self._best_record.score:
                    self._best_record = record

    def get_best(self) -> Optional[EvaluationRecord]:
        """Return the best evaluation record found so far."""
        return self._best_record

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Return the best parameters found so far."""
        if self._best_record is None:
            return None
        return self._best_record.params.copy()

    def get_best_score(self) -> Optional[float]:
        """Return the best score found so far."""
        if self._best_record is None:
            return None
        return self._best_record.score

    def get_all_scores(self) -> List[float]:
        """Return all scores in order."""
        return [record.score for record in self.records]

    def get_all_params(self) -> List[Dict[str, Any]]:
        """Return all parameters in order."""
        return [record.params for record in self.records]

    def size(self) -> int:
        """Return number of evaluations."""
        return len(self.records)

    def params_to_key(self, params: Dict[str, Any]) -> str:
        """
        Convert params dict to a hashable key for deduplication.
        Rounds continuous values to avoid floating point issues.
        """
        items = []
        for key in sorted(params.keys()):
            val = params[key]
            if isinstance(val, float):
                # Round to 6 decimal places to detect near-duplicates
                # (tolerance of ~1e-6 for parameter differences)
                val = round(val, 6)
            items.append((key, val))
        return str(tuple(items))

    def has_params(self, params: Dict[str, Any]) -> bool:
        """Check if parameters have already been evaluated."""
        key = self.params_to_key(params)
        return key in self._params_set

    def mark_evaluated(self, params: Dict[str, Any]) -> None:
        """Mark parameters as having been evaluated (without adding a record)."""
        key = self.params_to_key(params)
        self._params_set.add(key)

    def rebuild_params_set(self) -> None:
        """Rebuild the params set from records (useful if manually modified)."""
        self._params_set.clear()
        for record in self.records:
            key = self.params_to_key(record.params)
            self._params_set.add(key)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the optimization history."""
        scores = self.get_all_scores()
        if not scores:
            return {
                "n_evaluations": 0,
                "best_score": None,
                "mean_score": None,
                "std_score": None,
            }

        return {
            "n_evaluations": len(scores),
            "best_score": self._best_record.score,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

    def __repr__(self) -> str:
        best = self.get_best_score()
        return f"OptimizationHistory(n={self.size()}, best={best})"
