import pandas as pd
from pandas import DataFrame, Series, read_csv
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show, close
import matplotlib.pyplot as plt
import os

# --- Configuration ---
HEIGHT: int = 4
OUTPUT_DIR = "images/granularity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def get_variable_types(df: DataFrame) -> dict[str, list[str]]:
    """
    Identifies date and symbolic (categorical) variables.
    """
    variable_types: dict[str, list[str]] = {
        "date": [],
        "symbolic": []
    }
    
    # Detect dates based on column name keywords
    for col in df.select_dtypes(include=['datetime', 'object']).columns:
        try:
            if "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
                variable_types["date"].append(col)
        except:
            pass
            
    # Detect symbolic variables (excluding those already identified as dates)
    symbolic_cols = df.select_dtypes(include=['object', 'category']).columns
    variable_types["symbolic"] = [c for c in symbolic_cols if c not in variable_types["date"]]
    
    return variable_types

def plot_bar_chart(x: list, y: list, ax: plt.Axes, title: str, xlabel: str, ylabel: str, percentage: bool = False):
    """
    Helper function to plot bar charts.
    CHANGED: Removed the top-15 limit. Plots ALL categories as requested.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Plot ALL categories
    ax.bar(x, y, color="#4c72b0")
    
    # Rotate labels to 90 degrees to try and fit as many as possible
    ax.tick_params(axis='x', rotation=90, labelsize=8) 

# --- Functions from Professor's Example ---

def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    """
    Extracts year, quarter, month, day (and hour) from date columns.
    """
    for date in date_vars:
        df[date] = pd.to_datetime(df[date], errors='coerce')
        
        df[date + "_year"] = df[date].dt.year
        df[date + "_quarter"] = df[date].dt.quarter
        df[date + "_month"] = df[date].dt.month
        df[date + "_day"] = df[date].dt.day
        
        if df[date].dt.hour.nunique() > 1: 
            df[date + "_hour"] = df[date].dt.hour
            
    return df

def analyse_date_granularity(data: DataFrame, var: str, levels: list[str]) -> ndarray:
    """
    Plots granularity (distribution) for temporal variables.
    """
    valid_levels = [l for l in levels if var + "_" + l in data.columns]
    cols: int = len(valid_levels)
    fig: Figure
    axs: ndarray
    
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {var}")

    for i in range(cols):
        level_col = var + "_" + valid_levels[i]
        counts: Series[int] = data[level_col].value_counts().sort_index()
        
        plot_bar_chart(
            counts.index.astype(str).to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=valid_levels[i],
            xlabel=valid_levels[i],
            ylabel="nr records",
            percentage=False,
        )
    
    plt.tight_layout()
    return axs

def analyse_property_granularity(data: DataFrame, property: str, vars: list[str]) -> ndarray:
    """
    Plots granularity for symbolic/categorical variables.
    """
    cols: int = len(vars)
    fig: Figure
    axs: ndarray
    
    # Dynamic width: make chart wider if there are many variables
    fig, axs = subplots(1, cols, figsize=(cols * 6, HEIGHT), squeeze=False) # Increased width
    fig.suptitle(f"Granularity study for {property}")
    
    for i in range(cols):
        # Drop NA to avoid plotting 'nan'
        counts: Series[int] = data[vars[i]].value_counts().dropna()
        
        plot_bar_chart(
            counts.index.astype(str).to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=vars[i],
            xlabel=vars[i],
            ylabel="nr records",
            percentage=False,
        )
        
    plt.tight_layout()
    return axs

# --- Main Execution ---

# 1. Accidents Dataset
print("\n--- Processing Accidents ---")
file_tag_acc = "accidents"
try:
    data_acc: DataFrame = read_csv("data/raw/traffic_accidents.csv")
    date_cols = ['crash_date'] 
    data_acc = derive_date_variables(data_acc, date_cols)
    
    # Temporal
    for v_date in date_cols:
        analyse_date_granularity(data_acc, v_date, ["year", "quarter", "month", "day", "hour"])
        savefig(f"{OUTPUT_DIR}/{file_tag_acc}_granularity_{v_date}.png")
        print(f"   ✅ Saved temporal granularity for {v_date}")
        close()

    # Symbolic (All relevant variables)
    symbolic_vars = [
        'traffic_control_device', 'weather_condition', 'lighting_condition', 
        'first_crash_type', 'trafficway_type', 'alignment', 
        'roadway_surface_cond', 'road_defect', 'damage', 
        'prim_contributory_cause'
    ]
    # Check existence
    symbolic_vars = [v for v in symbolic_vars if v in data_acc.columns]

    for v_sym in symbolic_vars:
        analyse_property_granularity(data_acc, v_sym, [v_sym])
        savefig(f"{OUTPUT_DIR}/{file_tag_acc}_granularity_{v_sym}.png")
        print(f"   ✅ Saved symbolic granularity for {v_sym}")
        close()

except FileNotFoundError:
    print("❌ Error: 'data/raw/traffic_accidents.csv' not found.")


# 2. Flights Dataset
print("\n--- Processing Flights ---")
file_tag_flights = "flights"
try:
    # Sampling 5% for performance (plots will still be representative)
    data_flights: DataFrame = read_csv("data/raw/Combined_Flights_2022.csv").sample(frac=0.05, random_state=42)
    
    # Dates
    if 'FlightDate' in data_flights.columns:
        date_cols = ['FlightDate']
    else:
        date_cols = [col for col in data_flights.columns if 'Date' in col]

    if date_cols:
        data_flights = derive_date_variables(data_flights, date_cols)
        for v_date in date_cols:
            analyse_date_granularity(data_flights, v_date, ["year", "quarter", "month", "day"])
            savefig(f"{OUTPUT_DIR}/{file_tag_flights}_granularity_{v_date}.png")
            print(f"   ✅ Saved temporal granularity for {v_date}")
            close()
        
    # Symbolic - PLOT ALL (Removed limits)
    vars_types = get_variable_types(data_flights)
    
    # Only ignore FlightDate as it is handled above. 
    # We keep Tail_Number, Origin, etc., even if they have thousands of values.
    ignore = ['FlightDate'] 
    
    symbolic_vars = [v for v in vars_types['symbolic'] if v not in ignore]
    
    print(f"   ℹ️ Generating symbolic charts for {len(symbolic_vars)} variables (this may take a moment)...")

    for v_sym in symbolic_vars:
        try:
            analyse_property_granularity(data_flights, v_sym, [v_sym])
            savefig(f"{OUTPUT_DIR}/{file_tag_flights}_granularity_{v_sym}.png")
            print(f"   ✅ Saved symbolic granularity for {v_sym}")
            close()
        except Exception as e:
            print(f"   ⚠️ Could not plot {v_sym}: {e}")

except FileNotFoundError:
    print("❌ Error: 'data/raw/Combined_Flights_2022.csv' not found.")

print(f"\n✅ Done! Check folder {OUTPUT_DIR}")