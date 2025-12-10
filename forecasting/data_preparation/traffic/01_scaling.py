import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv, DataFrame, Series

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SCIENCE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, DATA_SCIENCE_ROOT)
sys.path.insert(0, os.path.join(DATA_SCIENCE_ROOT, "utils"))

from utils.dslabs_functions import plot_line_chart, HEIGHT, scale_all_dataframe

dataset_file = f"{DATA_SCIENCE_ROOT}/data/raw/TrafficTwoMonth.csv"
dataset_tag = "traffic_two_month"
target = "Total"
traffic_data = read_csv(dataset_file, sep=",", decimal=".")

# Creating proper datetime index
start_date = pd.Timestamp("2024-10-10 00:00:00")
traffic_data["datetime"] = pd.date_range(
    start=start_date, periods=len(traffic_data), freq="15min"
)
traffic_data = traffic_data.set_index("datetime")

# Drop non-numeric columns
traffic_data = traffic_data.drop(
    columns=["Date", "Time", "Day of the week", "Traffic Situation"]
)
series = traffic_data[target]


# Before scaling

plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{dataset_tag} {target} before scaling",
)
plt.tight_layout()
plt.savefig(f"images/{dataset_tag}_before_scaling.png")
plt.show() 
plt.clf()

# After scaling

df: DataFrame = scale_all_dataframe(traffic_data)

ss: Series = df[target]
plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    ss.index.to_list(),
    ss.to_list(),
    xlabel=ss.index.name,
    ylabel=target,
    title=f"{dataset_tag} {target} after scaling",
)
plt.tight_layout()
plt.savefig(f"images/{dataset_tag}_after_scaling.png")
plt.show()
plt.clf()

# Save scaled data
output_dir = os.path.join(SCRIPT_DIR, "processed_data")
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, f"{dataset_tag}_scaled.csv"))
