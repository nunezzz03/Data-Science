import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'))
from dslabs_functions import (
    ts_aggregation_by,
    plot_forecasting_series,
    plot_forecasting_eval,
    FORECAST_MEASURES
)
from lab5_config import TRAIN_TEST_SPLIT, LAG, AGGREGATION_CONFIGS

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images', 'aggregation')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'aggregation')
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class PersistenceModel:
    def __init__(self):
        self.last_value = None
    
    def fit(self, y_train):
        self.last_value = y_train.iloc[-1] if isinstance(y_train, pd.Series) else y_train[-1]
        return self
    
    def predict(self, X_test):
        n_samples = len(X_test) if hasattr(X_test, '__len__') else X_test.shape[0]
        return np.full(n_samples, self.last_value)


def prepare_supervised_data(series, lag=1):
    data = pd.DataFrame(series.values, columns=['value'])
    
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['value'].shift(i)
    
    data.dropna(inplace=True)
    
    X = data.drop('value', axis=1).values
    y = data['value'].values
    
    return X, y


def evaluate_models(train_series, test_series, config_name, lag=1):
    print(f"\n   Evaluating {config_name}...")
    
    X_train, y_train = prepare_supervised_data(train_series, lag=lag)
    X_test, y_test = prepare_supervised_data(test_series, lag=lag)
    
    results = {}
    persistence = PersistenceModel()
    persistence.fit(train_series)
    
    y_pred_persistence_train = persistence.predict(train_series)
    y_pred_persistence_test = persistence.predict(test_series)
    
    mse_pers = FORECAST_MEASURES['MSE'](test_series.values, y_pred_persistence_test)
    mae_pers = FORECAST_MEASURES['MAE'](test_series.values, y_pred_persistence_test)
    r2_pers = FORECAST_MEASURES['R2'](test_series.values, y_pred_persistence_test)
    
    results['Persistence'] = {
        'MSE': mse_pers,
        'MAE': mae_pers,
        'R2': r2_pers,
        'predictions_train': y_pred_persistence_train,
        'predictions_test': y_pred_persistence_test
    }
    
    print(f"      Persistence - MSE: {mse_pers:.4f}, MAE: {mae_pers:.4f}, R2: {r2_pers:.4f}")
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr_train = lr_model.predict(X_train)
    y_pred_lr_test = lr_model.predict(X_test)
    
    mse_lr = FORECAST_MEASURES['MSE'](y_test, y_pred_lr_test)
    mae_lr = FORECAST_MEASURES['MAE'](y_test, y_pred_lr_test)
    r2_lr = FORECAST_MEASURES['R2'](y_test, y_pred_lr_test)
    
    results['LinearRegression'] = {
        'MSE': mse_lr,
        'MAE': mae_lr,
        'R2': r2_lr,
        'predictions_train': y_pred_lr_train,
        'predictions_test': y_pred_lr_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    print(f"      LinearRegression - MSE: {mse_lr:.4f}, MAE: {mae_lr:.4f}, R2: {r2_lr:.4f}")
    
    return results


def run_aggregation_study(dataset_path, date_column, target_column, dataset_name):
    print(f"\n{'='*70}")
    print(f"AGGREGATION STUDY: {dataset_name}")
    print(f"{'='*70}")
    
    print(f"\n1. Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    if date_column == 'datetime' and 'Time' in df.columns and 'Date' in df.columns:
        df['datetime'] = pd.date_range(start='2023-01-10', periods=len(df), freq='15min')
        df.set_index('datetime', inplace=True)
    elif date_column is not None and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    
    if target_column not in df.columns:
        print(f"   Error: Target column '{target_column}' not found!")
        return
    
    series = df[target_column]
    print(f"   Loaded {len(series)} records")
    
    all_results = {}
    
    for config in AGGREGATION_CONFIGS:
        print(f"\n2. Testing Configuration: {config['name']}")
        print(f"   Granularity: {config['gran_level']}, Function: {config['agg_func']}")
        
        aggregated_series = ts_aggregation_by(
            series, 
            gran_level=config['gran_level'], 
            agg_func=config['agg_func']
        )
        print(f"   Aggregated to {len(aggregated_series)} records")
        
        trn_size = int(len(aggregated_series) * TRAIN_TEST_SPLIT)
        train_series = aggregated_series.iloc[:trn_size]
        test_series = aggregated_series.iloc[trn_size:]
        print(f"   Train: {len(train_series)} | Test: {len(test_series)}")
        
        results = evaluate_models(train_series, test_series, config['name'])
        all_results[config['name']] = {
            'metrics': results,
            'aggregated_series': aggregated_series
        }
        
        # Use professor's plot_forecasting_eval for Persistence model
        prd_trn_persistence = pd.Series(results['Persistence']['predictions_train'], index=train_series.index)
        prd_tst_persistence = pd.Series(results['Persistence']['predictions_test'], index=test_series.index)
        plot_forecasting_eval(
            train_series,
            test_series,
            prd_trn_persistence,
            prd_tst_persistence,
            title=f"Persistence Model Evaluation: {config['name']} - {dataset_name}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f"{dataset_name}_{config['name'].lower()}_persistence_eval.png"))
        plt.close()
        
        # Plot using professor's function - Persistence predictions
        prd_tst_persistence = pd.Series(results['Persistence']['predictions_test'], index=test_series.index)
        plot_forecasting_series(
            train_series, 
            test_series, 
            prd_tst_persistence,
            title=f"Aggregation: {config['name']} - {dataset_name}",
            xlabel='Time',
            ylabel=target_column
        )
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f"{dataset_name}_{config['name'].lower()}_persistence.png"))
        plt.close()
        
        train_aligned = train_series.iloc[1:]
        test_aligned = test_series.iloc[1:]
        prd_trn_lr = pd.Series(results['LinearRegression']['predictions_train'], index=train_aligned.index)
        prd_tst_lr = pd.Series(results['LinearRegression']['predictions_test'], index=test_aligned.index)
        
        # Use professor's plot_forecasting_eval for LinearRegression model
        plot_forecasting_eval(
            train_aligned,
            test_aligned,
            prd_trn_lr,
            prd_tst_lr,
            title=f"LinearRegression Model Evaluation: {config['name']} - {dataset_name}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f"{dataset_name}_{config['name'].lower()}_lr_eval.png"))
        plt.close()
        
        plot_forecasting_series(
            train_series,
            test_aligned,
            prd_tst_lr,
            title=f"Aggregation: {config['name']} - {dataset_name} (LinearRegression)",
            xlabel='Time',
            ylabel=target_column
        )
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f"{dataset_name}_{config['name'].lower()}_lr.png"))
        plt.close()
    
    print(f"\n{'='*70}")
    print("3. COMPARISON OF ALL CONFIGURATIONS")
    print(f"{'='*70}")
    
    comparison_data = []
    for config_name, result_data in all_results.items():
        for model_name, metrics in result_data['metrics'].items():
            comparison_data.append({
                'Configuration': config_name,
                'Model': model_name,
                'MSE': metrics['MSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    best_idx = comparison_df['R2'].idxmax()
    best_config = comparison_df.iloc[best_idx]
    
    print(f"\n{'='*70}")
    print("4. BEST CONFIGURATION SELECTED")
    print(f"{'='*70}")
    print(f"   Configuration: {best_config['Configuration']}")
    print(f"   Model: {best_config['Model']}")
    print(f"   MSE: {best_config['MSE']:.4f}")
    print(f"   MAE: {best_config['MAE']:.4f}")
    print(f"   R2: {best_config['R2']:.4f}")
    
    best_config_name = best_config['Configuration']
    best_aggregated_series = all_results[best_config_name]['aggregated_series']
    output_path = os.path.join(RESULTS_DIR, f"{dataset_name}_aggregated.csv")
    best_aggregated_series.to_csv(output_path, header=['value'])
    print(f"\n   Best aggregated dataset saved to: {output_path}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['MSE', 'MAE', 'R2']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = comparison_df.pivot(index='Configuration', columns='Model', values=metric)
        metric_data.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xlabel('Configuration')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"{dataset_name}_aggregation_summary.png"))
    plt.close()
    
    print(f"\n   Results saved to: {RESULTS_DIR}")
    print(f"   Visualizations saved to: {IMAGES_DIR}")
    
    return best_config


if __name__ == "__main__":
    run_aggregation_study(
        dataset_path='data/raw/economic_indicators_dataset_2010_2023.csv',
        date_column='Date',
        target_column='Inflation Rate (%)',
        dataset_name='economic'
    )
    
    run_aggregation_study(
        dataset_path='data/raw/TrafficTwoMonth.csv',
        date_column='datetime',
        target_column='Total',
        dataset_name='traffic'
    )
