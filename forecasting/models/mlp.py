"""
Lab 6 - Multi-Layer Perceptron (MLP) for Time Series Forecasting
Trains MLP models on univariate and multivariate time series data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from typing import Tuple, Dict, List, Any
import pickle
import warnings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import MLP_CONFIG, DATASETS, VIZ_DIR

warnings.filterwarnings('ignore')


class MLPTimeSeriesForecaster:
    """Multi-Layer Perceptron for time series forecasting."""
    
    def __init__(self, lookback: int = 12, horizon: int = 1):
        """
        Initialize MLP forecaster.
        
        Args:
            lookback: Number of past time steps to use as input (default: 12)
            horizon: Number of steps ahead to forecast (default: 1)
        """
        self.lookback = lookback
        self.horizon = horizon
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.best_params = None
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting.
        
        Args:
            data: Input time series data
            
        Returns:
            X: Input sequences of shape (n_samples, lookback, n_features)
            y: Target values of shape (n_samples, horizon)
        """
        X, y = [], []
        for i in range(len(data) - self.lookback - self.horizon + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.horizon])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple, hidden_layers: Tuple = (64, 32), 
                   activation: str = "relu", learning_rate: float = 0.001) -> models.Sequential:
        """
        Build MLP model architecture.
        
        Args:
            input_shape: Shape of input data (lookback, n_features)
            hidden_layers: Tuple of hidden layer sizes
            activation: Activation function
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential()
        
        # Flatten input if needed
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Flatten())
        
        # Add hidden layers
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(0.2))
        
        # Output layer
        output_units = self.horizon if input_shape[-1] == 1 else input_shape[-1] * self.horizon
        model.add(layers.Dense(output_units))
        
        # Compile
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              hidden_layers: Tuple = (64, 32), activation: str = "relu",
              learning_rate: float = 0.001, batch_size: int = 32,
              epochs: int = 100, patience: int = 15, verbose: int = 0) -> Dict:
        """
        Train MLP model with hyperparameters.
        
        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_val: Validation input sequences
            y_val: Validation targets
            hidden_layers: Hidden layer configuration
            activation: Activation function
            learning_rate: Learning rate
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Dictionary with training history and metrics
        """
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, hidden_layers, activation, learning_rate)
        
        # Early stopping callback
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, 
                                   restore_best_weights=True)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return {
            'history': self.history,
            'params': {
                'hidden_layers': hidden_layers,
                'activation': activation,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
            }
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test input sequences
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics
        """
        y_pred = self.predict(X_test)
        
        # Flatten for metric calculation
        y_test_flat = y_test.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        metrics = {
            'mse': mean_squared_error(y_test_flat, y_pred_flat),
            'mae': mean_absolute_error(y_test_flat, y_pred_flat),
            'r2': r2_score(y_test_flat, y_pred_flat),
        }
        
        return metrics
    
    def save_model(self, filepath: Path):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def grid_search_mlp(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   config: Dict, dataset_name: str, data_type: str) -> Dict:
    """
    Perform grid search over MLP hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        config: Hyperparameter configuration
        dataset_name: Name of dataset (traffic/economic)
        data_type: Data type (univariate/multivariate)
        
    Returns:
        Dictionary with best model, params, and results
    """
    config = config.get(data_type, {})
    
    # Hyperparameter combinations
    hidden_layers_list = config.get('hidden_layers', [(64, 32)])
    activations = config.get('activation', ['relu'])
    learning_rates = config.get('learning_rate', [0.001])
    batch_sizes = config.get('batch_size', [32])
    
    results = []
    best_r2 = float('-inf')
    best_model_info = None
    
    total_combinations = (len(hidden_layers_list) * len(activations) * 
                         len(learning_rates) * len(batch_sizes))
    
    print(f"\n{'='*80}")
    print(f"Grid Search: MLP - {dataset_name.upper()} ({data_type.upper()})")
    print(f"Total combinations: {total_combinations}")
    print(f"{'='*80}\n")
    
    combo_count = 0
    for hidden_layers, activation, lr, batch_size in product(
        hidden_layers_list, activations, learning_rates, batch_sizes
    ):
        combo_count += 1
        
        print(f"[{combo_count}/{total_combinations}] Testing: Hidden={hidden_layers}, "
              f"Activation={activation}, LR={lr}, BS={batch_size}")
        
        try:
            # Create and train model
            forecaster = MLPTimeSeriesForecaster(lookback=12, horizon=1)
            train_result = forecaster.train(
                X_train, y_train, X_val, y_val,
                hidden_layers=hidden_layers,
                activation=activation,
                learning_rate=lr,
                batch_size=batch_size,
                epochs=config.get('epochs', 100),
                patience=config.get('early_stopping_patience', 15),
                verbose=0
            )
            
            # Evaluate on test set
            metrics = forecaster.evaluate(X_test, y_test)
            
            # Store results
            result = {
                'hidden_layers': hidden_layers,
                'activation': activation,
                'learning_rate': lr,
                'batch_size': batch_size,
                'metrics': metrics,
                'model': forecaster,
                'train_epochs': len(train_result['history'].history['loss'])
            }
            results.append(result)
            
            print(f"    ✓ R²={metrics['r2']:8.4f} | MSE={metrics['mse']:8.4f} | MAE={metrics['mae']:8.4f} | Epochs={result['train_epochs']}")
            
            # Track best model
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model_info = result
                print(f"    ★ NEW BEST! R²={best_r2:.4f}")
        
        except Exception as e:
            print(f"    ✗ ERROR: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Best R²: {best_r2:.4f}")
    if best_model_info:
        best_params = {k: best_model_info[k] for k in 
                      ['hidden_layers', 'activation', 'learning_rate', 'batch_size']}
        print(f"Best params: {best_params}")
    print(f"{'='*80}\n")
    
    return {
        'best_model': best_model_info,
        'best_r2': best_r2,
        'all_results': results,
        'dataset': dataset_name,
        'data_type': data_type
    }


def save_metrics_to_csv(results: Dict, dataset_name: str, data_type: str):
    """Save best model metrics to CSV."""
    if not results['best_model']:
        return
    
    best = results['best_model']
    metrics_file = METRICS_DIR / f"mlp_{dataset_name}_{data_type}_metrics.csv"
    
    metrics_df = pd.DataFrame([{
        'technique': 'MLP',
        'dataset': dataset_name,
        'data_type': data_type,
        'hidden_layers': str(best['hidden_layers']),
        'activation': best['activation'],
        'learning_rate': best['learning_rate'],
        'batch_size': best['batch_size'],
        'train_epochs': best['train_epochs'],
        'mse': best['metrics']['mse'],
        'mae': best['metrics']['mae'],
        'r2': best['metrics']['r2'],
    }])
    
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")


def plot_hyperparameters_study(results: Dict, dataset_name: str, data_type: str):
    """Generate visualizations for hyperparameter studies."""
    all_results = results['all_results']
    
    if not all_results:
        return
    
    # Extract results by hyperparameter
    hidden_layers_r2 = {}
    activation_r2 = {}
    lr_r2 = {}
    bs_r2 = {}
    
    for res in all_results:
        hl_key = str(res['hidden_layers'])
        hidden_layers_r2.setdefault(hl_key, []).append(res['metrics']['r2'])
        
        activation_r2.setdefault(res['activation'], []).append(res['metrics']['r2'])
        lr_r2.setdefault(res['learning_rate'], []).append(res['metrics']['r2'])
        bs_r2.setdefault(res['batch_size'], []).append(res['metrics']['r2'])
    
    # Calculate averages
    hl_avg = {k: np.mean(v) for k, v in hidden_layers_r2.items()}
    act_avg = {k: np.mean(v) for k, v in activation_r2.items()}
    lr_avg = {k: np.mean(v) for k, v in lr_r2.items()}
    bs_avg = {k: np.mean(v) for k, v in bs_r2.items()}
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MLP Hyperparameter Study - {dataset_name.upper()} ({data_type.upper()})', 
                 fontsize=16, fontweight='bold')
    
    # Hidden layers
    axs[0, 0].bar(hl_avg.keys(), hl_avg.values(), color='steelblue')
    axs[0, 0].set_title('Hidden Layers Impact on R²')
    axs[0, 0].set_xlabel('Hidden Layers Configuration')
    axs[0, 0].set_ylabel('Average R²')
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # Activation
    axs[0, 1].bar(act_avg.keys(), act_avg.values(), color='darkorange')
    axs[0, 1].set_title('Activation Function Impact on R²')
    axs[0, 1].set_xlabel('Activation Function')
    axs[0, 1].set_ylabel('Average R²')
    
    # Learning rate
    lr_labels = [f'{lr:.4f}' for lr in lr_avg.keys()]
    axs[1, 0].bar(lr_labels, lr_avg.values(), color='seagreen')
    axs[1, 0].set_title('Learning Rate Impact on R²')
    axs[1, 0].set_xlabel('Learning Rate')
    axs[1, 0].set_ylabel('Average R²')
    
    # Batch size
    bs_labels = [str(bs) for bs in bs_avg.keys()]
    axs[1, 1].bar(bs_labels, bs_avg.values(), color='crimson')
    axs[1, 1].set_title('Batch Size Impact on R²')
    axs[1, 1].set_xlabel('Batch Size')
    axs[1, 1].set_ylabel('Average R²')
    
    plt.tight_layout()
    
    # Save figure
    output_file = VIZ_DIR / "hyperparameters" / f"mlp_{dataset_name}_{data_type}_hyperparam_study.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Hyperparameter study saved to {output_file}")
    plt.close()


def plot_best_predictions(y_test: np.ndarray, y_pred: np.ndarray, 
                         dataset_name: str, data_type: str):
    """Plot best model's predictions vs actual values."""
    y_test_flat = y_test.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = np.arange(len(y_test_flat))
    ax.plot(x_axis, y_test_flat, 'o-', label='Actual', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(x_axis, y_pred_flat, 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
    
    ax.set_title(f'Best MLP Model: Actual vs Predicted - {dataset_name.upper()} ({data_type.upper()})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = VIZ_DIR / "predictions" / f"mlp_{dataset_name}_{data_type}_predictions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Predictions plot saved to {output_file}")
    plt.close()


def plot_performance_comparison(results: Dict, dataset_name: str, data_type: str):
    """Plot best model performance metrics."""
    if not results['best_model']:
        return
    
    best = results['best_model']
    metrics = best['metrics']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Best MLP Model Performance - {dataset_name.upper()} ({data_type.upper()})',
                 fontsize=14, fontweight='bold')
    
    # Key metrics: MSE, MAE, R²
    metric_names = ['MSE', 'MAE', 'R²']
    metric_values = [metrics['mse'], metrics['mae'], metrics['r2']]
    colors = ['steelblue', 'darkorange', 'crimson']
    ax.bar(metric_names, metric_values, color=colors)
    ax.set_title('Performance Metrics', fontweight='bold')
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = VIZ_DIR / "performance" / f"mlp_{dataset_name}_{data_type}_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to {output_file}")
    plt.close()

def train_and_evaluate_mlp(dataset_name: str = 'traffic', 
                          data_type: str = 'univariate') -> Dict:
    """
    Main function to train and evaluate MLP models.
    
    Args:
        dataset_name: Name of dataset (traffic/economic)
        data_type: Data type (univariate/multivariate)
        
    Returns:
        Dictionary with training results
    """
    # Load prepared data from smoothing step
    dataset_config = DATASETS[dataset_name]
    data_path = dataset_config['data_path']
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading prepared data from: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Split into train/test (80/20 temporal split)
    split_idx = int(0.8 * len(data))
    X_train = data.iloc[:split_idx].values
    X_test = data.iloc[split_idx:].values
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Determine lookback based on test set size
    # Need at least (lookback + 1) samples in test set to create sequences
    lookback = 12
    if len(X_test) <= lookback:
        lookback = max(1, len(X_test) - 2)  # Leave at least 2 samples for test
        print(f"  ⚠ Dataset too small, reducing lookback to {lookback}")
    
    # Create sequences
    forecaster = MLPTimeSeriesForecaster(lookback=lookback, horizon=1)
    X_train_seq, y_train_seq = forecaster.create_sequences(X_train_scaled)
    X_test_seq, y_test_seq = forecaster.create_sequences(X_test_scaled)
    
    # Split into train/val
    split_idx = int(0.8 * len(X_train_seq))
    X_train_final = X_train_seq[:split_idx]
    y_train_final = y_train_seq[:split_idx]
    X_val = X_train_seq[split_idx:]
    y_val = y_train_seq[split_idx:]
    
    # Grid search
    results = grid_search_mlp(
        X_train_final, y_train_final,
        X_val, y_val,
        X_test_seq, y_test_seq,
        MLP_CONFIG, dataset_name, data_type
    )
    
    # Generate visualizations (charts only)
    if results['best_model']:
        # Get predictions from best model
        y_pred = results['best_model']['model'].predict(X_test_seq)
        
        plot_hyperparameters_study(results, dataset_name, data_type)
        plot_best_predictions(y_test_seq, y_pred, dataset_name, data_type)
        plot_performance_comparison(results, dataset_name, data_type)
    
    return results


if __name__ == "__main__":
    # Train univariate models only (both datasets are univariate)
    results_traffic_uni = train_and_evaluate_mlp('traffic', 'univariate')
    results_economic_uni = train_and_evaluate_mlp('economic', 'univariate')
