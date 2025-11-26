import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Style Configuration
sns.set_style("whitegrid")
OUTPUT_DIR = "images/profiling/granularity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_temporal_granularity(df, time_col, dataset_name):
    """Generates temporal granularity histograms (Year, Month, Day, Hour)"""
    print(f"   ⏳ Analyzing Temporal Granularity for {dataset_name}...")
    
    # Ensure datetime format
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        print(f"      Error converting {time_col}: {e}")
        return

    # Create temporal features
    df['Year'] = df[time_col].dt.year
    df['Month'] = df[time_col].dt.month_name()
    df['DayOfWeek'] = df[time_col].dt.day_name()
    df['Hour'] = df[time_col].dt.hour

    # Define correct order for days and months
    week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']

    # Create figure with subplots (Granularity)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Temporal Granularity: {dataset_name} ({time_col})', fontsize=16)

    # 1. Year
    sns.countplot(data=df, x='Year', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Records per Year')

    # 2. Month
    sns.countplot(data=df, x='Month', ax=axes[0,1], order=[m for m in month_order if m in df['Month'].unique()], color='salmon')
    axes[0,1].set_title('Records per Month')
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Day of Week
    sns.countplot(data=df, x='DayOfWeek', ax=axes[1,0], order=week_order, color='lightgreen')
    axes[1,0].set_title('Records per Day of Week')
    axes[1,0].tick_params(axis='x', rotation=45)

    # 4. Hour
    sns.countplot(data=df, x='Hour', ax=axes[1,1], color='gold')
    axes[1,1].set_title('Records per Hour')

    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/{dataset_name}_temporal_granularity.png"
    plt.savefig(filename)
    plt.close()
    print(f"      ✅ Chart saved: {filename}")

def plot_symbolic_granularity(df, dataset_name, top_n=15):
    """Generates histograms for symbolic (categorical) variables"""
    print(f"   🔠 Analyzing Symbolic Granularity for {dataset_name}...")
    
    # Select only text (object) or categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Ignore columns that are just IDs or Dates converted to text
    ignore_cols = ['Date', 'Time', 'FlightDate', 'Total', 'City'] # City sometimes has too many categories
    cat_cols = [c for c in cat_cols if c not in ignore_cols and "Date" not in c]

    if len(cat_cols) == 0:
        print("      ⚠️ No relevant symbolic variable found.")
        return

    for col in cat_cols:
        # Check if there are many unique categories (high cardinality)
        n_unique = df[col].nunique()
        
        if n_unique > 50:
            # If too many, show only Top N (Fine Granularity)
            plot_type = f"Top {top_n}"
            data_to_plot = df[col].value_counts().nlargest(top_n).reset_index()
            data_to_plot.columns = [col, 'Count']
        else:
            # If few, show all (Coarse Granularity)
            plot_type = "All"
            data_to_plot = df[col].value_counts().reset_index()
            data_to_plot.columns = [col, 'Count']

        plt.figure(figsize=(10, 6))
        sns.barplot(data=data_to_plot, y=col, x='Count', palette='viridis')
        plt.title(f'Symbolic Granularity: {dataset_name} - {col} ({plot_type})')
        plt.xlabel('Number of Records')
        
        filename = f"{OUTPUT_DIR}/{dataset_name}_symbolic_{col}.png"
        plt.savefig(filename)
        plt.close()
    
    print(f"      ✅ Symbolic charts saved in {OUTPUT_DIR}")

# --- EXECUTION ---

# 1. Traffic Dataset
print("\n--- Processing Traffic ---")
df_traffic = pd.read_csv('data/raw/TrafficTwoMonth.csv')
# Traffic has 'Time' column
plot_temporal_granularity(df_traffic, 'Time', 'Traffic')
# Traffic has no relevant symbolic columns (only numeric), skipping.

# 2. Accidents Dataset
print("\n--- Processing Accidents ---")
df_accidents = pd.read_csv('data/raw/traffic_accidents.csv')
plot_temporal_granularity(df_accidents, 'Date', 'Accidents')
plot_symbolic_granularity(df_accidents, 'Accidents')

# 3. Flights Dataset (Sampling for speed)
print("\n--- Processing Flights ---")
df_flights = pd.read_csv('data/raw/Combined_Flights_2022.csv').sample(frac=0.05, random_state=42)
plot_temporal_granularity(df_flights, 'FlightDate', 'Flights')
plot_symbolic_granularity(df_flights, 'Flights')

# 4. Economic Dataset
print("\n--- Processing Economic ---")
df_eco = pd.read_csv('data/raw/economic_indicators_dataset_2010_2023.csv')
# Economic 'Year' is numeric but represents time
df_eco['Date_Fake'] = pd.to_datetime(df_eco['Year'], format='%Y') # Trick to use temporal function
plot_temporal_granularity(df_eco, 'Date_Fake', 'Economic')
plot_symbolic_granularity(df_eco, 'Economic')

print(f"\n✅ Done! Check folder {OUTPUT_DIR}")