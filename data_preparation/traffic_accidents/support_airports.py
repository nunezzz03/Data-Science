import pandas as pd

# Load CSV
df = pd.read_csv(r"C:\Users\joesi\Desktop\Data Science\Data-Science\data\raw\Combined_Flights_2022.csv")

# Step 1: Check for duplicates
total_duplicates = df.duplicated().sum()
print(f"Total duplicated rows: {total_duplicates}")

# Step 2: Check unique IDs
unique_origins = df['OriginAirportID'].nunique()
unique_destinations = df['DestAirportID'].nunique()
print(f"Unique OriginAirportID: {unique_origins}")
print(f"Unique DestAirportID: {unique_destinations}")

# Step 3: Compare number of flights per airport
origin_counts = df.groupby('OriginAirportID').size()
dest_counts = df.groupby('DestAirportID').size()

# Airports with identical counts
identical_counts = (origin_counts.reindex(dest_counts.index).fillna(0) == dest_counts).sum()
print(f"Number of airports with identical flight counts as origin and destination: {identical_counts}")

# Step 4: Compare actual cancellation rates per airport
origin_cancel_rate = df.groupby('OriginAirportID')['Cancelled'].mean()
dest_cancel_rate = df.groupby('DestAirportID')['Cancelled'].mean()

# Merge into one DataFrame for comparison
comparison = pd.DataFrame({
    'Origin_Cancel_Rate': origin_cancel_rate,
    'Dest_Cancel_Rate': dest_cancel_rate.reindex(origin_cancel_rate.index)
})

comparison['Equal'] = comparison['Origin_Cancel_Rate'] == comparison['Dest_Cancel_Rate']

# Summary
print(f"Number of airports where cancellation rates are exactly equal: {comparison['Equal'].sum()}")
print(comparison.head(10))  # sample output