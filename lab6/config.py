"""
Lab 6 - Time Series Forecasting Models
Configuration file for hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
FORECASTING_ROOT = Path(__file__).parent.parent / "forecasting"
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ_DIR = RESULTS_DIR
REPORT_DIR = PROJECT_ROOT / "report"

# Prepared data paths (from data preparation smoothing step)
TRAFFIC_DATA = FORECASTING_ROOT / "data_preparation" / "traffic" / "processed_data" / "smoothing" / "traffic_smoothed.csv"
ECONOMIC_DATA = FORECASTING_ROOT / "data_preparation" / "economic" / "processed_data" / "smoothing" / "economic_usa_smoothed.csv"

# Datasets
DATASETS = {
    "traffic": {
        "data_path": TRAFFIC_DATA,
    },
    "economic": {
        "data_path": ECONOMIC_DATA,
    }
}

# Data configurations
DATA_TYPES = ["univariate", "multivariate"]

# ============================================================================
# EXPONENTIAL SMOOTHING HYPERPARAMETERS
# ============================================================================
EXPONENTIAL_SMOOTHING_CONFIG = {
    "enabled": True,
    "univariate": {
        "seasonal_periods": [12, 24, None],
        "trend": ["add", "mul", None],
        "seasonal": ["add", "mul", None],
        "initialization_method": ["estimated", "heuristic"],
    },
    "multivariate": {
        "seasonal_periods": [12, 24, None],
        "trend": ["add", "mul", None],
        "seasonal": ["add", "mul", None],
    }
}

# ============================================================================
# MLP HYPERPARAMETERS
# ============================================================================
MLP_CONFIG = {
    "enabled": True,
    "univariate": {
        "hidden_layers": [
            (64,),
            (128, 64),
        ],
        "activation": ["relu"],
        "learning_rate": [0.001],
        "batch_size": [32],
        "epochs": 100,
        "early_stopping_patience": 15,
        "validation_split": 0.2,
    },
    "multivariate": {
        "hidden_layers": [
            (64,),
            (128, 64),
        ],
        "activation": ["relu"],
        "learning_rate": [0.001],
        "batch_size": [32],
        "epochs": 100,
        "early_stopping_patience": 15,
        "validation_split": 0.2,
    }
}

# ============================================================================
# ARIMA HYPERPARAMETERS
# ============================================================================
ARIMA_CONFIG = {
    "enabled": True,
    "univariate": {
        "p": [0, 1, 2],
        "d": [0, 1, 2],
        "q": [0, 1, 2],
    },
    "multivariate": {
        "enabled": False,  # ARIMA is univariate only
    }
}

# ============================================================================
# LSTM HYPERPARAMETERS
# ============================================================================
LSTM_CONFIG = {
    "enabled": True,
    "univariate": {
        "units": [32, 64, 128],
        "layers": [1, 2],
        "dropout": [0.0, 0.2],
        "activation": ["relu", "tanh"],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [16, 32],
        "epochs": 100,
        "early_stopping_patience": 15,
        "validation_split": 0.2,
        "lookback": [12, 24],
    },
    "multivariate": {
        "units": [32, 64, 128],
        "layers": [1, 2],
        "dropout": [0.0, 0.2],
        "activation": ["relu", "tanh"],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [16, 32],
        "epochs": 100,
        "early_stopping_patience": 15,
        "validation_split": 0.2,
        "lookback": [12, 24],
    }
}

# ============================================================================
# GENERAL TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "verbose": True,
    "save_models": True,
    "save_metrics": True,
    "save_visualizations": True,
}

# ============================================================================
# METRICS TO CALCULATE
# ============================================================================
METRICS = ["mse", "mae", "rmse", "r2", "mape"]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VIZ_CONFIG = {
    "figsize": (12, 6),
    "dpi": 300,
    "style": "seaborn-v0_8-darkgrid",
    "font_size": 10,
}

# Create directories if they don't exist
def create_dirs():
    """Create all necessary directories for the project."""
    dirs = [
        VIZ_DIR / "hyperparameters",
        VIZ_DIR / "predictions",
        VIZ_DIR / "performance",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

# Create directories on import
create_dirs()
