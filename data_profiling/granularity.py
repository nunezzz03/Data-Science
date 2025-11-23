import pandas as pd
from pandas import DataFrame, Series, read_csv
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show, close
import matplotlib.pyplot as plt
import os

# Get paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration ---
HEIGHT: int = 5
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images", "granularity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def plot_bar_chart(x: list, y: list, ax: plt.Axes, title: str, xlabel: str, ylabel: str, percentage: bool = False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar(x, y, color="#4c72b0")
    ax.tick_params(axis="x", rotation=90, labelsize=8)

def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    for date in date_vars:
        df[date] = pd.to_datetime(df[date], errors="coerce")
        df[date + "_year"] = df[date].dt.year
        df[date + "_quarter"] = df[date].dt.quarter
        df[date + "_month"] = df[date].dt.month
        df[date + "_day"] = df[date].dt.day
        if df[date].dt.hour.nunique() > 1:
            df[date + "_hour"] = df[date].dt.hour
    return df

# --- MAPPING DICTIONARIES & FUNCTIONS ---

def map_weather_hierarchy(condition):
    c = str(condition).upper()
    if 'CLEAR' in c: return 'Good'
    if 'CLOUDY' in c or 'OVERCAST' in c: return 'Fair'
    if 'RAIN' in c or 'DRIZZLE' in c: return 'Rain'
    if 'SNOW' in c or 'SLEET' in c or 'HAIL' in c or 'FREEZING' in c: return 'Winter Precip.'
    if 'FOG' in c or 'SMOKE' in c or 'HAZE' in c or 'BLOWING' in c: return 'Low Visibility'
    return 'Other/Unknown'

def map_luminosity_hierarchy(condition):
    c = str(condition).upper()
    if 'DAYLIGHT' in c: return 'Day'
    if 'DARKNESS' in c: return 'Night'
    if 'DAWN' in c or 'DUSK' in c: return 'Twilight'
    return 'Unknown'

def map_traffic_control_hierarchy(device):
    d = str(device).upper()
    if 'NO CONTROLS' in d: return 'None'
    if 'SIGNAL' in d or 'FLASHER' in d: return 'Signal'
    if 'SIGN' in d: return 'Sign'
    if 'LANE' in d or 'MARKING' in d: return 'Marking'
    if 'OFFICER' in d or 'GUARD' in d: return 'Officer'
    return 'Other/Unknown'

def map_trafficway_hierarchy(way):
    w = str(way).upper()
    if 'NOT DIVIDED' in w: return 'Not Divided'
    if 'DIVIDED' in w: return 'Divided'
    if 'ONE-WAY' in w: return 'One-Way'
    if 'RAMP' in w: return 'Ramp'
    return 'Other'

def map_road_surface_hierarchy(surface):
    s = str(surface).upper()
    if 'DRY' in s: return 'Dry'
    if 'WET' in s: return 'Wet'
    if 'SNOW' in s or 'ICE' in s or 'SLUSH' in s: return 'Winter Cond.'
    return 'Other/Unknown'

def map_road_defect_hierarchy(defect):
    d = str(defect).upper()
    if 'NO DEFECTS' in d: return 'No Defects'
    if 'UNKNOWN' in d: return 'Unknown'
    return 'Defect Present'

def map_airline_name_hierarchy(airline):
    a = str(airline).upper()
    if any(x in a for x in ['AMERICAN', 'DELTA', 'UNITED', 'ALASKA', 'HAWAIIAN']): return 'Legacy'
    if any(x in a for x in ['SOUTHWEST', 'JETBLUE', 'SPIRIT', 'FRONTIER', 'ALLEGIANT']): return 'Low-Cost'
    if 'CARGO' in a: return 'Cargo'
    return 'Regional/Other'

def map_airline_code_hierarchy(code):
    # Codes for major US airlines (2022 data)
    c = str(code).upper()
    if c in ['AA', 'DL', 'UA', 'AS', 'HA']: return 'Legacy'
    if c in ['WN', 'B6', 'NK', 'F9', 'G4']: return 'Low-Cost'
    return 'Regional/Other'

def map_us_region(state_code):
    northeast = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']
    midwest = ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']
    south = ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX']
    west = ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
    if state_code in northeast: return 'Northeast'
    if state_code in midwest: return 'Midwest'
    if state_code in south: return 'South'
    if state_code in west: return 'West'
    return 'Other'

def map_tail_number_hierarchy(tail):
    t = str(tail).upper()
    if pd.isna(tail) or t == 'NAN': return 'Unknown'
    if t.startswith('N'): return 'USA (N-Prefix)'
    return 'Foreign/Other'


def derive_symbolic_hierarchies(df: DataFrame, dataset_name: str) -> DataFrame:
    """Applies hierarchies to the dataframe columns"""
    if dataset_name == "accidents":
        mappings = {
            'weather_condition': ('weather_type', map_weather_hierarchy),
            'lighting_condition': ('day_phase', map_luminosity_hierarchy),
            'traffic_control_device': ('traffic_control_group', map_traffic_control_hierarchy),
            'trafficway_type': ('trafficway_group', map_trafficway_hierarchy),
            'roadway_surface_cond': ('surface_group', map_road_surface_hierarchy),
            'road_defect': ('defect_group', map_road_defect_hierarchy)
        }
    elif dataset_name == "flights":
        mappings = {
            'Airline': ('airline_type', map_airline_name_hierarchy),
            'Marketing_Airline_Network': ('marketing_airline_type', map_airline_code_hierarchy),
            'Operating_Airline': ('operating_airline_type', map_airline_name_hierarchy),
            'IATA_Code_Marketing_Airline': ('iata_marketing_type', map_airline_code_hierarchy),
            'IATA_Code_Operating_Airline': ('iata_operating_type', map_airline_code_hierarchy),
            'OriginState': ('origin_region', map_us_region),
            'DestState': ('dest_region', map_us_region),
            'Tail_Number': ('registration_country', map_tail_number_hierarchy)
        }
    else:
        return df

    for col, (new_col, func) in mappings.items():
        if col in df.columns:
            df[new_col] = df[col].apply(func)
    return df


# --- Analysis Functions ---

def analyse_date_granularity(data: DataFrame, var: str, levels: list[str]) -> ndarray:
    valid_levels = [l for l in levels if var + "_" + l in data.columns]
    cols = len(valid_levels)
    fig, axs = subplots(1, cols, figsize=(cols * 4, HEIGHT), squeeze=False)
    fig.suptitle(f"Temporal Granularity: {var}")
    for i in range(cols):
        counts = data[var + "_" + valid_levels[i]].value_counts().sort_index()
        plot_bar_chart(counts.index.astype(str), counts.values, axs[0, i], valid_levels[i], valid_levels[i], "nr records")
    plt.tight_layout()
    return axs

def analyse_hierarchy_granularity(data: DataFrame, original_var: str, hierarchy_var: str) -> ndarray:
    if original_var not in data.columns or hierarchy_var not in data.columns:
        return None
    
    fig, axs = subplots(1, 2, figsize=(12, HEIGHT), squeeze=False)
    fig.suptitle(f"Hierarchy: {original_var} -> {hierarchy_var}")

    # Plot 1: Hierarchy (Grouped)
    counts_h = data[hierarchy_var].value_counts()
    plot_bar_chart(counts_h.index.astype(str), counts_h.values, axs[0, 0], f"Group: {hierarchy_var}", hierarchy_var, "nr records")

    # Plot 2: Original (Detailed) - Limit to top 30 for readability
    counts_o = data[original_var].value_counts()
    if len(counts_o) > 30:
        counts_o = counts_o.head(30)
        suffix = " (Top 30)"
    else:
        suffix = ""
    plot_bar_chart(counts_o.index.astype(str), counts_o.values, axs[0, 1], f"Original: {original_var}{suffix}", original_var, "nr records")
    
    plt.tight_layout()
    return axs


# --- Main Execution ---

# 1. Accidents
print("--- Processing Accidents ---")
try:
    df_acc = read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "traffic_accidents.csv"))
    df_acc = derive_date_variables(df_acc, ["crash_date"])
    df_acc = derive_symbolic_hierarchies(df_acc, "accidents")

    # Temporal
    analyse_date_granularity(df_acc, "crash_date", ["year", "month", "day", "hour"])
    savefig(f"{OUTPUT_DIR}/accidents_granularity_date.png")
    close()

    # Symbolic Hierarchies
    pairs = [
        ("weather_condition", "weather_type"),
        ("lighting_condition", "day_phase"),
        ("traffic_control_device", "traffic_control_group"),
        ("trafficway_type", "trafficway_group"),
        ("roadway_surface_cond", "surface_group"),
        ("road_defect", "defect_group")
    ]
    for orig, hier in pairs:
        analyse_hierarchy_granularity(df_acc, orig, hier)
        savefig(f"{OUTPUT_DIR}/accidents_granularity_{orig}_hierarchy.png")
        close()
    print("   ✅ Accidents plots saved.")
except FileNotFoundError:
    print("❌ Error: traffic_accidents.csv not found.")

# 2. Flights
print("\n--- Processing Flights ---")
try:
    df_fli = read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "Combined_Flights_2022.csv")).sample(frac=0.05, random_state=42)
    
    date_col = "FlightDate" if "FlightDate" in df_fli.columns else [c for c in df_fli.columns if "Date" in c][0]
    df_fli = derive_date_variables(df_fli, [date_col])
    df_fli = derive_symbolic_hierarchies(df_fli, "flights")

    # Temporal
    analyse_date_granularity(df_fli, date_col, ["year", "quarter", "month", "day"])
    savefig(f"{OUTPUT_DIR}/flights_granularity_date.png")
    close()

    # Symbolic Hierarchies
    # Special case for Cities: Hierarchy is State
    analyse_hierarchy_granularity(df_fli, "OriginCityName", "OriginState")
    savefig(f"{OUTPUT_DIR}/flights_granularity_OriginCity_hierarchy.png")
    close()
    
    analyse_hierarchy_granularity(df_fli, "DestCityName", "DestState")
    savefig(f"{OUTPUT_DIR}/flights_granularity_DestCity_hierarchy.png")
    close()

    # Standard Hierarchies
    pairs = [
        ("Airline", "airline_type"),
        ("Marketing_Airline_Network", "marketing_airline_type"),
        ("Operating_Airline", "operating_airline_type"),
        ("IATA_Code_Marketing_Airline", "iata_marketing_type"),
        ("IATA_Code_Operating_Airline", "iata_operating_type"),
        ("OriginState", "origin_region"),
        ("DestState", "dest_region"),
        ("Tail_Number", "registration_country")
    ]
    for orig, hier in pairs:
        analyse_hierarchy_granularity(df_fli, orig, hier)
        savefig(f"{OUTPUT_DIR}/flights_granularity_{orig}_hierarchy.png")
        close()
    print("   ✅ Flights plots saved.")
except FileNotFoundError:
    print("❌ Error: Combined_Flights_2022.csv not found.")

print(f"\n✅ Done! Check folder {OUTPUT_DIR}")