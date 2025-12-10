import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame, read_csv
from matplotlib.pyplot import figure, show, subplots, plot, legend, savefig
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from dslabs_functions import HEIGHT, set_chart_labels, plot_line_chart, plot_multiline_chart
from pathlib import Path
import os

# Configuração de Output
SCRIPT_DIR = Path(__file__).parent

# =============================================================================
# 1. HELPER FUNCTIONS 
# =============================================================================

def ts_aggregation_by(series: Series, gran_level: str = "D", agg_func: str = "sum") -> Series:
    """
    Agrega uma série temporal pela granularidade desejada.
    gran_level: 'D' (Daily), 'W' (Weekly), 'M' (Monthly), 'Q' (Quarterly)
    """
    df = series.to_frame()
    # Usa 'ME' para Month End nas versões recentes do Pandas, 'M' em antigas
    agg = df.resample(gran_level).agg(agg_func)
    return agg[series.name]

def plot_components(series: Series, title: str = "", x_label: str = "time", y_label: str = "") -> list[Axes]:
    """
    Plota a decomposição sazonal (Observed, Trend, Seasonal, Residual)
    """
    # model='add' (Aditivo) ou 'mul' (Multiplicativo). 
    # Para dados com zeros ou negativos, usar 'add'.
    decomposition: DecomposeResult = seasonal_decompose(series, model="add", period=get_period(series))
    
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    return axs

def get_period(series: Series) -> int:
    """Tenta adivinhar o periodo para decomposição baseado na frequência"""
    # Ajusta conforme os teus dados reais
    if series.index.freqstr:
        if 'h' in series.index.freqstr.lower(): return 24 # 24 horas
        if 'd' in series.index.freqstr.lower(): return 7  # 7 dias (semana)
        if 'm' in series.index.freqstr.lower(): return 12 # 12 meses
    return 12 # Default

def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series

# =============================================================================
# 2. ANALYSIS FUNCTIONS 
# =============================================================================

def analyze_distribution(series: Series, target: str, file_tag: str, output_dir: Path):
    print(f"--- Processing Distribution for {file_tag} ---")
    
    # 2.1 Aggregations for Distribution
    ss_days = ts_aggregation_by(series, gran_level="D", agg_func="sum")
    ss_weeks = ts_aggregation_by(series, gran_level="W", agg_func="sum")
    ss_months = ts_aggregation_by(series, gran_level="ME", agg_func="sum")

    grans = [series, ss_days, ss_weeks, ss_months]
    gran_names = ["Hourly", "Daily", "Weekly", "Monthly"]

    # 2.2 Histograms Matrix
    fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
    fig.suptitle(f"{file_tag} {target} Distribution")
    for i in range(len(grans)):
        set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
        axs[i].hist(grans[i].dropna().values)
    savefig(output_dir / f"{file_tag}_distribution_histograms.png")
    plt.close()  # ✅ ADICIONAR: Fecha a figura em vez de mostrar
    print(f"   📊 Saved: {output_dir / f'{file_tag}_distribution_histograms.png'}")

    # 2.3 Boxplots Comparison
    fig, axs = subplots(2, 2, figsize=(2 * HEIGHT, HEIGHT))
    set_chart_labels(axs[0, 0], title="HOURLY")
    axs[0, 0].boxplot(series.dropna())
    
    set_chart_labels(axs[0, 1], title="WEEKLY")
    axs[0, 1].boxplot(ss_weeks.dropna())
    
    axs[1, 0].grid(False); axs[1, 0].set_axis_off()
    axs[1, 0].text(0.2, 0, str(series.describe()), fontsize="small")
    
    axs[1, 1].grid(False); axs[1, 1].set_axis_off()
    axs[1, 1].text(0.2, 0, str(ss_weeks.describe()), fontsize="small")
    savefig(output_dir / f"{file_tag}_distribution_boxplots.png")
    plt.close()  # ✅ ADICIONAR
    print(f"   📊 Saved: {output_dir / f'{file_tag}_distribution_boxplots.png'}")
    
    # 2.4 Lag Plot
    figure(figsize=(3 * HEIGHT, HEIGHT))
    lags = get_lagged_series(series, 20, 10)
    plot_multiline_chart(series.index.to_list(), lags, xlabel="Timestamp", ylabel=target)
    savefig(output_dir / f"{file_tag}_lag_plot.png")
    plt.close()  # ✅ ADICIONAR
    print(f"   📊 Saved: {output_dir / f'{file_tag}_lag_plot.png'}")
    
    # 2.5 Autocorrelation
    autocorrelation_study(series, 10, 1, file_tag, output_dir)

def autocorrelation_study(series: Series, max_lag: int, delta: int = 1, file_tag: str = "", output_dir: Path = None):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series.dropna(), maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    savefig(output_dir / f"{file_tag}_autocorrelation.png")
    plt.close()  # ✅ ADICIONAR
    print(f"   📊 Saved: {output_dir / f'{file_tag}_autocorrelation.png'}")

def analyze_stationarity(series: Series, target: str, file_tag: str, output_dir: Path):
    print(f"--- Processing Stationarity for {file_tag} ---")
    
    # 3.1 Mean Line Plot
    n: int = len(series)
    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=target,
        title=f"{file_tag} stationary study",
        name="original",
    )
    plot(series.index, [series.mean()] * n, "r-", label="mean")
    legend()
    savefig(output_dir / f"{file_tag}_stationarity_mean.png")
    plt.close()  # ✅ ADICIONAR
    print(f"   📊 Saved: {output_dir / f'{file_tag}_stationarity_mean.png'}")

    # 3.2 Binned Mean (Estabilidade da média)
    BINS = 10
    mean_line: list[float] = []
    series_vals = series.dropna() # Evitar erros com NaNs
    n = len(series_vals)
    
    for i in range(BINS):
        segment = series_vals[i * n // BINS : (i + 1) * n // BINS]
        mean_value = [segment.mean()] * (n // BINS)
        mean_line += mean_value
    
    # Ajuste final de tamanho
    mean_line += [mean_line[-1]] * (n - len(mean_line))

    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        series_vals.index.to_list(),
        series_vals.to_list(),
        xlabel=series_vals.index.name,
        ylabel=target,
        title=f"{file_tag} stationary study (Binned Mean)",
        name="original",
        show_stdev=True,
    )
    plot(series_vals.index, mean_line, "r-", label="mean")
    legend()
    savefig(output_dir / f"{file_tag}_stationarity_binned.png")
    plt.close()  # ✅ ADICIONAR
    print(f"   📊 Saved: {output_dir / f'{file_tag}_stationarity_binned.png'}")

    # 3.3 ADF Test
    eval_stationarity(series_vals)

def eval_stationarity(series: Series) -> bool:
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    
    is_stationary = result[1] <= 0.05
    print(f"The series {('is' if is_stationary else 'is not')} stationary")
    return is_stationary

def analyze_seasonality(series: Series, target: str, file_tag: str, output_dir: Path):
    print(f"--- Processing Seasonality for {file_tag} ---")
    
    # 4.1 Plot Components
    try:
        plot_components(
            series,
            title=f"{file_tag} {target} decomposition",
            x_label="Time",
            y_label=target
        )
    except ValueError as e:
        print(f"Erro na decomposição direta (possíveis dados em falta): {e}")
        print("Tentando com agregação diária...")
        daily_series = series.resample('D').sum().fillna(0)
        plot_components(
            daily_series,
            title=f"{file_tag} {target} decomposition (Daily Agg)",
            x_label="Time",
            y_label=target
        )
        
    savefig(output_dir / f"{file_tag}_seasonality.png")
    plt.close()  # ✅ ADICIONAR
    print(f"   📊 Saved: {output_dir / f'{file_tag}_seasonality.png'}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Configuração dos datasets
    datasets = [
        {
            "filename": SCRIPT_DIR / "TrafficTwoMonth_processed.csv",
            "file_tag": "Traffic",
            "target_col": "Total",
            "output_dir": SCRIPT_DIR / "outputs" / "TrafficTwoMonth"
        },
        {
            "filename": SCRIPT_DIR / "EconomicUSA_processed.csv",
            "file_tag": "Economic",
            "target_col": "Inflation Rate (%)",
            "output_dir": SCRIPT_DIR / "outputs" / "EconomicUSA"
        }
    ]
    
    # Processar cada dataset
    for ds_config in datasets:
        print("\n" + "="*80)
        print(f"  PROCESSANDO: {ds_config['file_tag']}")
        print("="*80 + "\n")
        
        # Atualizar OUTPUT_DIR global para este dataset
        OUTPUT_DIR = ds_config['output_dir']
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        FILENAME = ds_config['filename']
        FILE_TAG = ds_config['file_tag']
        TARGET_COL = ds_config['target_col']
        
        # Carregar Dados
        if FILENAME.exists():
            print(f"📂 Carregando {FILENAME}...")
            
            try:
                # Carregar com parse_dates no índice
                data = read_csv(
                    FILENAME,
                    index_col='Timestamp' if FILE_TAG == "Traffic" else 'Date',  # Economic usa 'Date'
                    parse_dates=True
                )
                
                # Verificar se o índice é DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    print("❌ ERRO: Índice não é DatetimeIndex")
                    continue
                
                # Ordenar pelo tempo
                data = data.sort_index()
                
                print(f"✅ Dados carregados! Registos: {len(data)}")
                print(f"📊 Colunas: {data.columns.tolist()}")
                print(f"📅 Período: {data.index.min()} até {data.index.max()}")
                print(f"\n🔍 Primeiras linhas:")
                print(data.head())
                
                # Seleciona a coluna alvo
                series = data[TARGET_COL]
                
                # Executar Análises
                print(f"\n📁 Imagens serão guardadas em: {OUTPUT_DIR}\n")
                
                analyze_distribution(series, TARGET_COL, FILE_TAG, OUTPUT_DIR)
                analyze_stationarity(series, TARGET_COL, FILE_TAG, OUTPUT_DIR)
                analyze_seasonality(series, TARGET_COL, FILE_TAG, OUTPUT_DIR)
                
                print(f"\n✅ Análise de {FILE_TAG} completa!")
                print(f"📂 Todas as imagens em: {OUTPUT_DIR}")
                
            except Exception as e:
                print(f"\n❌ ERRO ao processar {FILE_TAG}: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print(f"❌ Ficheiro não encontrado: {FILENAME}")
            print("💡 Execute primeiro o Dimentionality_Granularity.py!")
    
    print("\n" + "="*80)
    print("  🎉 ANÁLISE COMPLETA DE TODOS OS DATASETS!")
    print("="*80)