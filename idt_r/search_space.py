"""
Search space handling for IDT-R optimizer.

Supports continuous, discrete (integer), and categorical hyperparameters.
"""

from typing import Dict, Tuple, List, Any, Union
import numpy as np


class Variable:
    """Base class for a hyperparameter variable."""

    def __init__(self, name: str):
        self.name = name

    def is_continuous(self) -> bool:
        raise NotImplementedError

    def is_discrete(self) -> bool:
        raise NotImplementedError

    def is_categorical(self) -> bool:
        raise NotImplementedError

    def denormalize(self, normalized_value: float) -> Any:
        """Convert a normalized [0, 1] value to actual parameter value."""
        raise NotImplementedError

    def normalize(self, value: Any) -> float:
        """Convert an actual parameter value to [0, 1] range."""
        raise NotImplementedError


class ContinuousVariable(Variable):
    """Continuous hyperparameter (float)."""

    def __init__(self, name: str, min_val: float, max_val: float):
        super().__init__(name)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        if self.min_val >= self.max_val:
            raise ValueError(f"min_val must be < max_val for {name}")

    def is_continuous(self) -> bool:
        return True

    def is_discrete(self) -> bool:
        return False

    def is_categorical(self) -> bool:
        return False

    def denormalize(self, normalized_value: float) -> float:
        return self.min_val + normalized_value * (self.max_val - self.min_val)

    def normalize(self, value: float) -> float:
        return (value - self.min_val) / (self.max_val - self.min_val)

    def __repr__(self) -> str:
        return f"Continuous({self.name}, [{self.min_val}, {self.max_val}])"


class DiscreteVariable(Variable):
    """Discrete hyperparameter (integer)."""

    def __init__(self, name: str, min_val: int, max_val: int):
        super().__init__(name)
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        if self.min_val >= self.max_val:
            raise ValueError(f"min_val must be < max_val for {name}")

    def is_continuous(self) -> bool:
        return False

    def is_discrete(self) -> bool:
        return True

    def is_categorical(self) -> bool:
        return False

    def denormalize(self, normalized_value: float) -> int:
        value = self.min_val + normalized_value * (self.max_val - self.min_val + 1)
        return int(np.clip(value, self.min_val, self.max_val))

    def normalize(self, value: int) -> float:
        return (value - self.min_val) / (self.max_val - self.min_val + 1)

    def __repr__(self) -> str:
        return f"Discrete({self.name}, [{self.min_val}, {self.max_val}])"


class CategoricalVariable(Variable):
    """Categorical hyperparameter."""

    def __init__(self, name: str, categories: List[Any]):
        super().__init__(name)
        self.categories = list(categories)
        if len(self.categories) == 0:
            raise ValueError(f"categories must not be empty for {name}")
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

    def is_continuous(self) -> bool:
        return False

    def is_discrete(self) -> bool:
        return False

    def is_categorical(self) -> bool:
        return True

    def denormalize(self, normalized_value: float) -> Any:
        idx = int(np.clip(normalized_value * len(self.categories), 0, len(self.categories) - 1))
        return self.categories[idx]

    def normalize(self, value: Any) -> float:
        if value not in self.category_to_idx:
            raise ValueError(f"Unknown category '{value}' for {self.name}")
        idx = self.category_to_idx[value]
        return (idx + 0.5) / len(self.categories)

    def __repr__(self) -> str:
        return f"Categorical({self.name}, {self.categories})"


class SearchSpace:
    """
    Manages the search space for hyperparameter optimization.
    
    Supports:
    - Continuous (float): specify as (min, max)
    - Discrete (integer): specify as (min, max) with type hint
    - Categorical: specify as list of values
    """

    def __init__(self, space_dict: Dict[str, Union[Tuple, List]]):
        """
        Initialize search space.
        
        Parameters:
        -----------
        space_dict : Dict[str, Union[Tuple, List]]
            Dictionary defining the search space. Examples:
            {
                "learning_rate": (0.001, 0.1),      # continuous
                "n_estimators": (10, 500),          # discrete (if int bounds)
                "max_depth": (3, 20),               # discrete (if int bounds)
                "criterion": ["gini", "entropy"],   # categorical
            }
        """
        self.variables: Dict[str, Variable] = {}
        self._param_names = []

        for name, spec in space_dict.items():
            if isinstance(spec, list):
                # Categorical (always a list, like ["A", "B", "C"])
                self.variables[name] = CategoricalVariable(name, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                min_val, max_val = spec
                # Determine if continuous or discrete based on type
                if isinstance(min_val, int) and isinstance(max_val, int):
                    self.variables[name] = DiscreteVariable(name, int(min_val), int(max_val))
                else:
                    self.variables[name] = ContinuousVariable(name, float(min_val), float(max_val))
            else:
                raise ValueError(f"Invalid spec for {name}: {spec}")

            self._param_names.append(name)

    def get_param_names(self) -> List[str]:
        """Return list of parameter names."""
        return self._param_names.copy()

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Return normalized bounds [0, 1] for all variables."""
        return [(0.0, 1.0) for _ in self._param_names]

    def denormalize(self, normalized_params: np.ndarray) -> Dict[str, Any]:
        """
        Convert normalized [0, 1] array to actual parameter dictionary.
        
        Parameters:
        -----------
        normalized_params : np.ndarray
            Normalized parameters in [0, 1] range
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of actual parameter values
        """
        result = {}
        for i, name in enumerate(self._param_names):
            result[name] = self.variables[name].denormalize(normalized_params[i])
        return result

    def normalize(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Convert actual parameters to normalized [0, 1] array.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Dictionary of actual parameter values
            
        Returns:
        --------
        np.ndarray
            Normalized parameters in [0, 1] range
        """
        normalized = []
        for name in self._param_names:
            if name not in params:
                raise ValueError(f"Missing parameter: {name}")
            normalized.append(self.variables[name].normalize(params[name]))
        return np.array(normalized, dtype=np.float32)

    def sample_uniform(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """Generate random samples from the search space."""
        samples = []
        for _ in range(n_samples):
            normalized = np.random.uniform(0, 1, len(self._param_names))
            sample = self.denormalize(normalized)
            samples.append(sample)
        return samples

    def sample_within_bounds(
        self,
        bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Sample uniformly from within specified parameter bounds (in normalized space).
        
        Parameters:
        -----------
        bounds : Dict[str, Tuple[float, float]]
            Lower and upper bounds for each parameter in normalized [0, 1] space
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of parameter dictionaries sampled within bounds
        """
        samples = []
        for _ in range(n_samples):
            normalized = np.zeros(len(self._param_names))
            for i, name in enumerate(self._param_names):
                if name in bounds:
                    lower, upper = bounds[name]
                    normalized[i] = np.random.uniform(lower, upper)
                else:
                    normalized[i] = np.random.uniform(0, 1)
            sample = self.denormalize(normalized)
            samples.append(sample)
        return samples

    def __repr__(self) -> str:
        var_strs = [str(self.variables[name]) for name in self._param_names]
        return f"SearchSpace({', '.join(var_strs)})"
