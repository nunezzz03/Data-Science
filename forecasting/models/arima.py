# =============================================================================
# ARIMA FORECASTING - Using Smoothed Data
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig, subplots, close
from statsmodels.tsa.arima.model import ARIMA
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
FORECASTING_ROOT = SCRIPT_DIR.parent
DATA_SCIENCE_ROOT = FORECASTING_ROOT.parent

# ✅ Usar data/prepared (mesma estrutura que exponential_smoothing.py)
TRAFFIC_SMOOTHED = DATA_SCIENCE_ROOT / "data" / "prepared" / "traffic_smoothed.csv"
ECONOMIC_SMOOTHED = DATA_SCIENCE_ROOT / "data" / "prepared" / "economic_usa_smoothed.csv"

IMAGES_DIR = FORECASTING_ROOT / "outputs" / "models"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Adicionar utils ao path
sys.path.insert(0, str(DATA_SCIENCE_ROOT))
sys.path.insert(0, str(DATA_SCIENCE_ROOT / "utils"))

# Imports das funções auxiliares
try:
    from utils.dslabs_functions import (
        series_train_test_split, 
        HEIGHT, 
        FORECAST_MEASURES, 
        DELTA_IMPROVE,
        plot_multiline_chart,
        plot_forecasting_series
    )
except ImportError:
    print("⚠️  dslabs_functions not found. Using custom implementations...")
    HEIGHT = 4
    DELTA_IMPROVE = 0.001
    
    def series_train_test_split(data: Series, trn_pct: float = 0.9):
        """Split time series into train and test"""
        split_idx = int(len(data) * trn_pct)
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    def plot_multiline_chart(x, y_dict, ax=None, title="", xlabel="", ylabel="", percentage=False):
        """Plot multiple lines on same chart"""
        if ax is None:
            from matplotlib.pyplot import figure
            figure(figsize=(HEIGHT*2, HEIGHT))
            ax = None
        
        for label, values in y_dict.items():
            if ax is not None:
                ax.plot(x, values, marker='o', label=f'q={label}')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                from matplotlib.pyplot import plot, title as plt_title, xlabel as plt_xlabel, ylabel as plt_ylabel, legend, grid
                plot(x, values, marker='o', label=f'q={label}')
    
    def plot_forecasting_series(train, test, forecast, title="", xlabel="", ylabel=""):
        """Plot train, test and forecast series"""
        from matplotlib.pyplot import figure, plot, legend, title as plt_title, xlabel as plt_xlabel, ylabel as plt_ylabel, grid
        
        figure(figsize=(3*HEIGHT, HEIGHT))
        plot(train.index, train.values, label='Train', color='blue')
        plot(test.index, test.values, label='Test', color='green')
        plot(test.index, forecast, label='Forecast', color='red', linestyle='--')
        plt_title(title)
        plt_xlabel(xlabel)
        plt_ylabel(ylabel)
        legend()
        grid(True, alpha=0.3)
    
    # Forecast measures
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = np.abs(y_true) > 0.1  # Avoid division by near-zero
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    FORECAST_MEASURES = {
        "R2": r2_score,
        "MAE": lambda y, p: -mean_absolute_error(y, p),  # Negative because we maximize
        "RMSE": lambda y, p: -np.sqrt(mean_squared_error(y, p)),
        "MAPE": lambda y, p: -mape(y, p)
    }

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASETS = {
    "traffic": {
        "path": TRAFFIC_SMOOTHED,
        "target": "Total",  # ✅ Mesma coluna que exponential_smoothing.py
        "file_tag": "traffic",
        "measure": "R2"
    },
    "economic": {
        "path": ECONOMIC_SMOOTHED,
        "target": "Inflation Rate (%)",  # ✅ Mesma coluna que exponential_smoothing.py
        "file_tag": "economic",
        "measure": "R2"
    }
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data(dataset_name: str) -> Series:
    """Load dataset by name (traffic or economic)"""
    config = DATASETS[dataset_name]
    
    print(f"\n📂 Loading {dataset_name} data from: {config['path']}")
    
    if not config['path'].exists():
        raise FileNotFoundError(
            f"Dataset not found: {config['path']}\n"
            f"Available files in data/prepared/:\n" +
            "\n".join(f"  - {f.name}" for f in (DATA_SCIENCE_ROOT / "data" / "prepared").glob("*.csv"))
        )
    
    # Carregar CSV - primeira coluna deve ser datetime (index)
    data = read_csv(
        config['path'],
        index_col=0,
        parse_dates=True
    )
    
    # ✅ Auto-detect target se não existir
    if config['target'] not in data.columns:
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in {config['path']}")
        
        actual_target = numeric_cols[0]
        print(f"   ⚠️  '{config['target']}' not found. Using '{actual_target}' instead.")
        config['target'] = actual_target
    
    series = data[config['target']]
    
    print(f"   ✅ Loaded {len(series)} observations")
    print(f"   📅 Period: {series.index.min()} to {series.index.max()}")
    print(f"   🎯 Target: {config['target']}")
    
    return series

# ✅ WRAPPER para fix do bug da biblioteca
def safe_series_train_test_split(series: Series, trn_pct: float = 0.9):
    """
    Wrapper para series_train_test_split que funciona com Series.
    A função original do dslabs_functions tem um bug com Series.
    """
    # Converter Series para DataFrame temporariamente
    df_temp = series.to_frame()
    
    # Usar a função da biblioteca
    train_df, test_df = series_train_test_split(df_temp, trn_pct=trn_pct)
    
    # ✅ Verificar se já são Series ou se são DataFrames
    if isinstance(train_df, Series):
        # Já é Series, retornar direto
        return train_df, test_df
    else:
        # É DataFrame, extrair a primeira coluna
        train = train_df.iloc[:, 0]
        test = test_df.iloc[:, 0]
        return train, test

# =============================================================================
# ARIMA STUDY FUNCTION
# =============================================================================

def arima_study(train: Series, test: Series, file_tag: str, measure: str = "R2"):
    """
    Grid search for ARIMA parameters with visualization
    Following the professor's template
    """
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        
        for q in q_params:
            yvalues = []
            for p in p_params:
                try:
                    arima = ARIMA(train, order=(p, d, q))
                    model = arima.fit()
                    prd_tst = model.forecast(steps=len(test))
                    eval_score: float = FORECAST_MEASURES[measure](test, prd_tst)
                    
                    print(f"   ARIMA ({p}, {d}, {q}) - {measure}: {eval_score:.4f}")
                    
                    if eval_score > best_performance and abs(eval_score - best_performance) > DELTA_IMPROVE:
                        best_performance = eval_score
                        best_params["params"] = (p, d, q)
                        best_model = model
                    
                    yvalues.append(eval_score)
                    
                except Exception as e:
                    print(f"   ⚠️  ARIMA ({p}, {d}, {q}) failed: {str(e)}")
                    yvalues.append(best_performance if best_performance != -100000 else 0)
            
            values[q] = yvalues
        
        plot_multiline_chart(
            p_params, 
            values, 
            ax=axs[i], 
            title=f"ARIMA d={d} ({measure})", 
            xlabel="p", 
            ylabel=measure, 
            percentage=flag
        )
    
    print(f"\n✅ ARIMA best results achieved with (p,d,q)={best_params['params']} ==> measure={best_performance:.2f}")
    
    savefig(IMAGES_DIR / f"{file_tag}_arima_{measure}_study.png", dpi=150)
    print(f"   📊 Saved: {IMAGES_DIR / f'{file_tag}_arima_{measure}_study.png'}")
    close()
    
    return best_model, best_params

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_arima_analysis(dataset_name: str):
    """
    Run complete ARIMA analysis for a dataset
    """
    print("\n" + "="*70)
    print(f"ARIMA FORECASTING: {dataset_name.upper()}")
    print("="*70)
    
    config = DATASETS[dataset_name]
    file_tag = config['file_tag']
    measure = config['measure']
    
    # Load data
    series = load_data(dataset_name)
    
    # Train/Test split (90/10) - ✅ Usar a função wrapper
    train, test = safe_series_train_test_split(series, trn_pct=0.90)
    print(f"\n📊 Train/Test Split (90/10):")
    print(f"   Train: {len(train)} records")
    print(f"   Test:  {len(test)} records")
    
    # Initial ARIMA model (3,1,2)
    print(f"\n🔧 Fitting initial ARIMA(3, 1, 2)...")
    predictor = ARIMA(train, order=(3, 1, 2))
    model = predictor.fit()
    print(model.summary())
    
    # Plot diagnostics
    print(f"\n📊 Generating diagnostic plots...")
    model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
    savefig(IMAGES_DIR / f"{file_tag}_arima_diagnostics.png", dpi=150)
    print(f"   📊 Saved: {IMAGES_DIR / f'{file_tag}_arima_diagnostics.png'}")
    close()
    
    # ARIMA parameter study
    print(f"\n🔍 Running ARIMA parameter study with {measure}...")
    best_model, best_params = arima_study(train, test, file_tag, measure)
    
    # Generate forecast
    print(f"\n🔮 Generating forecast with best model...")
    prd_tst = best_model.forecast(steps=len(test))
    
    # Plot forecast
    print(f"\n📊 Plotting forecast results...")
    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - ARIMA {best_params['params']}",
        xlabel="Time",
        ylabel=config['target'],
    )
    savefig(IMAGES_DIR / f"{file_tag}_arima_{measure}_forecast.png", dpi=150)
    print(f"   📊 Saved: {IMAGES_DIR / f'{file_tag}_arima_{measure}_forecast.png'}")
    close()
    
    # Evaluate
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    mae = mean_absolute_error(test, prd_tst)
    rmse = np.sqrt(mean_squared_error(test, prd_tst))
    r2 = FORECAST_MEASURES["R2"](test, prd_tst)
    
    print(f"\n📊 Final Model Performance:")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    print(f"\n✅ {dataset_name.upper()} ARIMA analysis complete!")
    
    return best_model, best_params

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("LAB 6 - ARIMA FORECASTING")
    print("=" * 70)
    
    results = {}
    
    for dataset_name in ["traffic", "economic"]:
        try:
            best_model, best_params = run_arima_analysis(dataset_name)
            results[dataset_name] = {
                "model": best_model,
                "params": best_params
            }
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Images saved to: {IMAGES_DIR}")
    print("=" * 70)