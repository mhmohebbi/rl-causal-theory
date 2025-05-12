"""
Constants and configuration setting in the project.
Default hyperparameters and model registry mappings are defined here.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Type
import numpy as np
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy.stats import loguniform, randint

# -----------------------------------------------------------------------------
# MLPRegressor default hyper‑parameters
# -----------------------------------------------------------------------------
MLP_PARAMS: Dict[str, Any] = {
    "hidden_layer_sizes": (64, 32, 16),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 4000,
    "batch_size": 32,
    "learning_rate": "adaptive",
    "learning_rate_init": 0.001,
    "tol": 1e-4,
    "n_iter_no_change": 50,
    "alpha": 1e-4,
    "early_stopping": True,
    "validation_fraction": 0.1,
}

MLP_PARAM_SPACE = {
    # width^depth grid: every tuple is one architecture
    "hidden_layer_sizes": [
        (256, 128, 64),
        (128, 64, 32),
        (64, 32, 16),
        (128, 64),
        (64, 32),
        (128,),
    ],
    "activation": ["relu", "tanh"],
    "alpha":      loguniform(1e-6, 1e-3),       # L2
    "learning_rate_init": loguniform(1e-4, 1e-2),
    "batch_size": randint(32, 257),             # 32-256
    "solver": ["adam", "sgd"],
    "momentum":   loguniform(0.5, 0.99),        # only read when solver="sgd"
}

MLP_FIT_KWARGS = {}

# -----------------------------------------------------------------------------
# XGBRegressor default hyper‑parameters
# -----------------------------------------------------------------------------
XGB_PARAMS: Dict[str, Any] = {
    "tree_method": "hist",
    "learning_rate": 0.05,
    "n_estimators": 4000,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

XGB_PARAM_SPACE: Dict[str, Any] = {
    "learning_rate":  np.logspace(-3, -1, 20),   # 0.001 – 0.1
    "max_depth":      [3, 4, 5, 6, 7, 8],
    "min_child_weight":[0.5, 1, 3, 5, 10],
    "subsample":      [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":[0.5, 0.7, 0.9, 1.0],
    "gamma":          [0, 0.1, 0.2, 0.5, 1],
    "reg_alpha":      np.logspace(-3, 1, 10),    # 0.001 – 10
    "reg_lambda":     np.logspace(-3, 1, 10),
}

XGB_FIT_KWARGS = {
    "verbose": False,
}

# -----------------------------------------------------------------------------
# Registry - maps a human‑readable name to (estimator class, params) pair.
# -----------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Tuple[Type, Dict[str, Any]]] = {
    "mlp": (MLPRegressor, MLP_PARAMS, MLP_PARAM_SPACE, MLP_FIT_KWARGS),
    "xgboost": (XGBRegressor, XGB_PARAMS, XGB_PARAM_SPACE, XGB_FIT_KWARGS),
}

__all__ = [
    "MLP_PARAMS",
    "XGB_PARAMS",
    "MODEL_REGISTRY",
]
