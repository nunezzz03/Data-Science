import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(r"C:\Users\joesi\Desktop\Data Science\Data-Science\data\raw\Combined_Flights_2022.csv")
# Filter rows where Cancelled is True
# cancelled_flights = df[df['Cancelled'] == True]
# print(cancelled_flights[['Cancelled', 'OriginAirportID']].head(10))

# Force pandas to print all rows
pd.set_option('display.max_rows', None)  # show all rows
pd.set_option('display.max_columns', None)  # optional: show all columns
pd.set_option('display.width', None)  # prevent line wrapping

# Compute cancellation rate per airline
cancellation_rates = df.groupby('Operating_Airline')['Cancelled'].mean().sort_values(ascending=False)

# Convert to percentage
cancellation_rates_percent = cancellation_rates * 100

# Turn into DataFrame for nice display
cancellation_df = cancellation_rates_percent.reset_index()
cancellation_df.columns = ['Operating_Airline', 'Cancellation Rate (%)']
print(cancellation_df)


# Optional: bar chart
cancellation_df.plot(x='Operating_Airline', y='Cancellation Rate (%)', kind='bar', figsize=(10,6), color='skyblue')
plt.ylabel("Cancellation Rate (%)")
plt.title("Cancellation Rate per Airline")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()