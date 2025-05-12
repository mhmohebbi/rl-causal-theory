"""baselines.py
A simple, extensible wrapper around common regression baselines.
Usage
-----
>>> from baselines import BaselineRegressor
>>> model = BaselineRegressor("xgboost").train(X_train, y_train)
>>> rmse = model.evaluate(X_test, y_test)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from const import MODEL_REGISTRY  # maps name -> (cls, params)

# -----------------------------------------------------------------------------
# Abstract base class
# -----------------------------------------------------------------------------
class BaseEstimator(ABC):
    """Interface every baseline implementation must satisfy."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> "BaseEstimator":
        """Fit the underlying model and return self for chaining."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return predictions for *X*."""

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, *, metric: str = "rmse") -> float:
        """Evaluate *X* / *y* with the chosen *metric* and return a scalar score."""

    @abstractmethod
    def train_and_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_distributions: Dict[str, Any],
        *,
        cv: int = 3,
        n_iter: int = 25,
        scoring: str | None = None,
    ) -> Dict[str, Any]:
        """Hyper‑parameter search that updates the model in place.

        Returns the best parameter set found.
        """


# -----------------------------------------------------------------------------
# Concrete implementation: BaselineRegressor
# -----------------------------------------------------------------------------
class BaselineRegressor(BaseEstimator):
    """Factory driven baseline that wraps an arbitrary underlying regressor.
    
    This class provides a standardized interface to various regression models,
    allowing for consistent model creation, training, evaluation, and hyperparameter tuning.
    
    Attributes:
        model: The underlying regression model instance.
        name: The name of the regression model (lowercase).
    """

    def __init__(self, model_name: str, **override_params: Any) -> None:  # noqa: D401
        """Initialize a regression model with the specified name and parameters.
        
        Args:
            model_name: Name of the regression model to use (case-insensitive).
            **override_params: Parameters that override the default parameters for the model.
            
        Raises:
            ValueError: If the specified model name is not found in MODEL_REGISTRY.
        """
        try:
            # Look up the model class and default parameters from the registry
            cls, default_params, param_space, fit_kwargs = MODEL_REGISTRY[model_name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown baseline '{model_name}'. Available: {list(MODEL_REGISTRY)}") from exc

        # Merge default parameters with any overrides provided
        merged_params = {**default_params, **override_params}

        # Instantiate the model with the merged parameters
        self.model = cls(**merged_params)  # type: ignore[arg-type]
        self.param_space = param_space
        self.fit_params = fit_kwargs
        self.name = model_name.lower()
        self.best_iteration_ = None

    # ------------------------------------------------------------------
    # API implementation
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> "BaselineRegressor":
        """Train the regression model on the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            
        Returns:
            Self, to allow for method chaining.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return predictions for the input data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predicted values of shape (n_samples,).
        """
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, *, metric: str = "rmse") -> float:
        """Evaluate model performance on the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True target values of shape (n_samples,).
            metric: Evaluation metric to use. Options: "rmse", "mse", "r2".
            
        Returns:
            Scalar performance score (higher is better for r2, lower is better for rmse/mse).
            
        Raises:
            ValueError: If an unsupported metric is specified.
        """
        preds = self.predict(X)
        metric = metric.lower()
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y, preds)))
        elif metric == "mse":
            return float(mean_squared_error(y, preds))
        elif metric == "r2":
            return float(r2_score(y, preds))
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")

    def train_and_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cv: int = 5,
        n_iter: int = 100,
        scoring: str | None = None,
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning using randomized search cross-validation.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            param_distributions: Dictionary with parameters names as keys and distributions
                                as values for the randomized search.
            cv: Number of cross-validation folds.
            n_iter: Number of parameter settings sampled.
            scoring: Scoring metric to use. If None, uses neg_root_mean_squared_error 
                    (or neg_mean_squared_error for MLP).
                    
        Returns:
            Dictionary with the best parameters found.
        """
        if self.param_space is None:
            raise ValueError("No parameter space defined for this model.")

        # Use appropriate scoring metric based on model type
        scoring = scoring or ("neg_root_mean_squared_error" if self.name != "mlp" else "neg_mean_squared_error")

        # special case for XGBoost
        if self.name == "xgboost":
            # carve %10 of the data for eval set
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1, random_state=42)
            self.fit_params["eval_set"] = [(X_eval, y_eval)]
            self.model.set_params(early_stopping_rounds=100)
            X, y = X_train, y_train

        # Configure and run randomized search
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,  # Use all available CPU cores
            verbose=1,   # Show progress information
            random_state=42,  # For reproducibility
            refit=True,
            scoring=scoring,
        )
        search.fit(X, y, **(self.fit_params or {}))
        # Update the model with the best estimator found
        self.model = search.best_estimator_
        self.best_iteration_ = getattr(self.model, "best_iteration_", None)
        return search.best_params_

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # noqa: D401
        """Return the parameters of the underlying model.
        
        Args:
            deep: If True, will return the parameters for this estimator and
                 contained subobjects that are estimators.
                 
        Returns:
            Dictionary of parameter names mapped to their values.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "BaselineRegressor":
        """Set the parameters of the underlying model.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            Self, to allow for method chaining.
        """
        self.model.set_params(**params)
        return self

    # Representation helpers
    def __repr__(self) -> str:  # noqa: D401
        """Return a string representation of the BaselineRegressor.
        
        Returns:
            String representation showing class name and the underlying model.
        """
        return f"{self.__class__.__name__}(model={self.model!r})"


__all__: Tuple[str, ...] = ("BaselineRegressor",)
