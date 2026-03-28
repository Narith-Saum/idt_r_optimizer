"""
Utility functions for IDT-R optimizer.
"""

from typing import List, Dict, Any, Callable
import numpy as np


def remove_duplicate_params(
    params_list: List[Dict[str, Any]],
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """
    Remove duplicate parameter sets from a list.
    
    Parameters:
    -----------
    params_list : List[Dict[str, Any]]
        List of parameter dictionaries
    tolerance : float
        Tolerance for comparing float values
        
    Returns:
    --------
    List[Dict[str, Any]]
        List with duplicates removed
    """
    unique = []
    seen = set()

    for params in params_list:
        key = _params_to_key(params, tolerance)
        if key not in seen:
            seen.add(key)
            unique.append(params)

    return unique


def _params_to_key(params: Dict[str, Any], tolerance: float = 1e-8) -> str:
    """Convert params dict to hashable key."""
    items = []
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, float):
            # Round float to tolerance precision
            val = round(val / tolerance) * tolerance
        items.append((key, str(val)))
    return "|".join([f"{k}={v}" for k, v in items])


def filter_new_params(
    candidates: List[Dict[str, Any]],
    evaluated_params: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Filter out candidates that have already been evaluated.
    
    Parameters:
    -----------
    candidates : List[Dict[str, Any]]
        Newly proposed candidates
    evaluated_params : List[Dict[str, Any]]
        Previously evaluated parameters
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of candidates not yet evaluated
    """
    evaluated_keys = set(_params_to_key(p) for p in evaluated_params)
    new_candidates = [
        c for c in candidates if _params_to_key(c) not in evaluated_keys
    ]
    return new_candidates


def compute_feature_importance(
    tree_predictions: np.ndarray,
    y_actual: np.ndarray,
) -> Dict[str, float]:
    """
    Compute simple feature importance based on prediction error correlation.
    
    Parameters:
    -----------
    tree_predictions : np.ndarray
        Predictions from tree on training data
    y_actual : np.ndarray
        Actual objective values
        
    Returns:
    --------
    Dict[str, float]
        Feature importance scores (placeholder for tree feature importance)
    """
    # This is a placeholder - real implementation would extract from sklearn tree
    residuals = np.abs(y_actual - tree_predictions)
    mse = np.mean(residuals ** 2)
    return {"mse_train": float(mse)}


def normalize_scores(
    scores: List[float],
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize a list of scores to [0, 1] range.
    
    Parameters:
    -----------
    scores : List[float]
        Raw scores
    method : str
        Normalization method: 'minmax' or 'zscore'
        
    Returns:
    --------
    np.ndarray
        Normalized scores
    """
    scores_arr = np.array(scores)

    if method == "minmax":
        min_val = np.min(scores_arr)
        max_val = np.max(scores_arr)
        if max_val == min_val:
            return np.ones_like(scores_arr) * 0.5
        return (scores_arr - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean_val = np.mean(scores_arr)
        std_val = np.std(scores_arr)
        if std_val == 0:
            return np.zeros_like(scores_arr)
        normalized = (scores_arr - mean_val) / std_val
        # Clip to reasonable range
        return np.clip(normalized, -3, 3) / 3 + 0.5

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def format_params_for_display(params: Dict[str, Any]) -> str:
    """Format parameters nicely for logging."""
    items = []
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, float):
            items.append(f"{key}={val:.6g}")
        else:
            items.append(f"{key}={val}")
    return "{" + ", ".join(items) + "}"


def estimate_search_space_size(search_space_dict: Dict[str, Any]) -> float:
    """
    Estimate the size of the search space.
    
    For continuous: infinity
    For discrete: product of ranges
    For categorical: product of option counts
    """
    size = 1.0

    for name, spec in search_space_dict.items():
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            min_val, max_val = spec
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Discrete
                size *= (max_val - min_val + 1)
            else:
                # Continuous
                return float("inf")
        elif isinstance(spec, list):
            # Categorical
            size *= len(spec)

    return size
