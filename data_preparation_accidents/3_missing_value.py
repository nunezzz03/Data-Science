from numpy import ndarray
from pandas import DataFrame, read_csv
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN
from pandas import read_csv
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types
import numpy as np
from numpy import ndarray
from sklearn.impute import SimpleImputer
from dslabs_functions import get_variable_types
from scipy.stats import skew
import matplotlib.pyplot as plt
# Load dataset
df = read_csv("data/raw/traffic_accidents.csv")

print(f"Original File: {len(df)} records")
#df = df.replace("UNKNOWN", 2)
df = df.replace("UNKNOWN", np.nan)
#df = df[(df != "UNKNOWN").all(axis=1)]
print(f"File after MV removal: {len(df)} records")

#print(df['crash_month'].unique())

# Remove target-leakage columns
leak_cols = [
    "first_crash_type", "damage", "most_severe_injury", "injuries_total",
    "injuries_fatal", "injuries_incapacitating", "injuries_non_incapacitating",
    "injuries_reported_not_evident", "injuries_no_indication", "num_units", "prim_contributory_cause"
]
df = df.drop(columns=leak_cols, errors="ignore")


# Define encoding mappings
# Ordinary and Binary encodings
weather_condition_values: dict[str, int] = {
    "CLEAR": 1,
    "CLOUDY/OVERCAST": 1,
    "OTHER": 1,
    "FOG/SMOKE/HAZE": 2,
    "RAIN": 2,
    "BLOWING SAND, SOIL, DIRT": 2,
    "SNOW": 3,
    "SLEET/HAIL": 3,
    "FREEZING RAIN/DRIZZLE": 4,
    "BLOWING SNOW": 4,
    "SEVERE CROSS WIND GATE": 4
}
lighting_condition_values: dict[str, int] = {
    "DAYLIGHT": 1,
    "DAWN": 2,
    "DUSK": 2,
    "DARKNESS, LIGHTED ROAD": 3,
    "DARKNESS": 4
}
trafficway_type_values: dict[str, int] = {
    "NOT REPORTED": 1,
    "ALLEY": 1,
    "DRIVEWAY": 1,
    "PARKING LOT": 1,
    "ONE-WAY": 1,
    "NOT DIVIDED": 1,
    "DIVIDED - W/MEDIAN (NOT RAISED)": 2,
    "DIVIDED - W/MEDIAN BARRIER": 2,
    "CENTER TURN LANE": 2,
    "T-INTERSECTION": 3,
    "L-INTERSECTION": 3,
    "FOUR WAY": 3,
    "Y-INTERSECTION": 3,
    "RAMP": 3,
    "TRAFFIC ROUTE": 3,
    "ROUNDABOUT": 4,
    "FIVE POINT, OR MORE": 4,
    "OTHER": 2,
    "UNKNOWN INTERSECTION TYPE": 2
}
roadway_surface_cond_values: dict[str, int] = {
    "DRY": 1,
    "OTHER": 2,
    "WET": 2,
    "SAND, MUD, DIRT": 3,
    "SNOW OR SLUSH": 3,
    "ICE": 4
}
road_defect_values: dict[str, int] = {
    "NO DEFECTS": 1,
    "OTHER": 2,
    "SHOULDER DEFECT": 2,
    "WORN SURFACE": 3,
    "DEBRIS ON ROADWAY": 3,
    "RUT, HOLES": 4
}
crash_type_values: dict[str, int] = {
    "NO INJURY / DRIVE AWAY": 1,
    "INJURY AND / OR TOW DUE TO CRASH": 2
}
intersection_related_i_values: dict[str, int] = {
    "N": 0,
    "Y": 1
}
prim_contributory_cause_values: dict[str, int] = {
    "NOT APPLICABLE": 1,
    "UNABLE TO DETERMINE": 1,
    "IMPROPER TURNING/NO SIGNAL": 2,
    "FOLLOWING TOO CLOSELY": 2,
    "IMPROPER BACKING": 2,
    "IMPROPER OVERTAKING/PASSING": 2,
    "DRIVING ON WRONG SIDE/WRONG WAY": 2,
    "IMPROPER LANE USAGE": 2,
    "DISREGARDING TRAFFIC SIGNALS": 3,
    "DRIVING SKILLS/KNOWLEDGE/EXPERIENCE": 3,
    "FAILING TO REDUCE SPEED TO AVOID CRASH": 3,
    "DISREGARDING STOP SIGN": 3,
    "OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER": 3,
    "DISREGARDING OTHER TRAFFIC SIGNS": 3,
    "VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)": 3,
    "DISTRACTION - FROM OUTSIDE VEHICLE": 3,
    "DISTRACTION - FROM INSIDE VEHICLE": 3,
    "DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)": 3,
    "UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)": 3,
    "EXCEEDING SAFE SPEED FOR CONDITIONS": 3,
    "EXCEEDING AUTHORIZED SPEED LIMIT": 3,
    "ROAD ENGINEERING/SURFACE/MARKING DEFECTS": 4,
    "DISREGARDING ROAD MARKINGS": 4,
    "PHYSICAL CONDITION OF DRIVER": 4,
    "ROAD CONSTRUCTION/MAINTENANCE": 4,
    "WEATHER": 4,
    "EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST": 4,
    "ANIMAL": 4,
    "CELL PHONE USE OTHER THAN TEXTING": 4,
    "HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)": 4,
    "RELATED TO BUS STOP": 4,
    "TEXTING": 4,
    "OBSTRUCTED CROSSWALKS": 4,
    "DISREGARDING YIELD SIGN": 4,
    "MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT": 4,
    "BICYCLE ADVANCING LEGALLY ON RED LIGHT": 4,
    "PASSING STOPPED SCHOOL BUS": 4
}


# Mapping for ordinal and binary encoding
encoding: dict[str, dict[str, int]] = {
    "weather_condition": weather_condition_values,
    "lighting_condition": lighting_condition_values,
    "trafficway_type": trafficway_type_values,
    "roadway_surface_cond": roadway_surface_cond_values,
    "road_defect": road_defect_values,
    "crash_type": crash_type_values,
    "intersection_related_i": intersection_related_i_values,
    "prim_contributory_cause": prim_contributory_cause_values,
    "crash_type": crash_type_values
}

# Apply encoding
df = df.replace(encoding, inplace=False)

# Cyclical encodings
df['crash_hour_sin'] = np.sin(2 * np.pi * df['crash_hour'] / 24) + 1
df['crash_hour_cos'] = np.cos(2 * np.pi * df['crash_hour'] / 24) + 1
df['crash_day_sin'] = np.sin(2 * np.pi * df['crash_day_of_week'] / 7) + 1
df['crash_day_cos'] = np.cos(2 * np.pi * df['crash_day_of_week'] / 7) + 1
df['crash_month_sin'] = np.sin(2 * np.pi * df['crash_month'] / 12) + 1
df['crash_month_cos'] = np.cos(2 * np.pi * df['crash_month'] / 12) + 1

#drop original columns(features)
#df = df.drop(columns=['crash_day_of_week'])


def mvi_by_filling_adaptive(data: DataFrame, strategy: str = "frequent") -> DataFrame:
    df: DataFrame
    variables: dict = get_variable_types(data)
    stg_sym, v_sym = "most_frequent", "NA"
    stg_bool, v_bool = "most_frequent", False
    lst_dfs: list = []

    # Numeric columns – adaptive
    if len(variables["numeric"]) > 0:
        df_num = data[variables["numeric"]].copy()
        for col in df_num.columns:
            # Choose median if skewed, else mean
            col_skew = skew(df_num[col].dropna())
            fill_value = df_num[col].median() if abs(col_skew) > 1 else df_num[col].mean()
            df_num[col] = df_num[col].fillna(fill_value)
        lst_dfs.append(df_num)

    # Symbolic columns
    if len(variables["symbolic"]) > 0:
        imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
        tmp_sb = DataFrame(
            imp.fit_transform(data[variables["symbolic"]]),
            columns=variables["symbolic"],
        )
        lst_dfs.append(tmp_sb)

    # Binary columns
    if len(variables["binary"]) > 0:
        imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
        tmp_bool = DataFrame(
            imp.fit_transform(data[variables["binary"]]),
            columns=variables["binary"],
        )
        lst_dfs.append(tmp_bool)

    df = pd.concat(lst_dfs, axis=1)
    return df

df=mvi_by_filling_adaptive(df)

# Keep numeric + binary columns
numeric_df = df.select_dtypes(include=["number"])
binary_cols = [col for col in df.select_dtypes(include=["object"]).columns if df[col].nunique()==2]
binary_df = df[binary_cols].apply(lambda c: c.astype("category").cat.codes)

# Combine
df = pd.concat([numeric_df, binary_df], axis=1)

# Split 70/30
train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)

# Save
train_df.to_csv("data_preparation_accidents/train_datasets/traffic_accidents_train_3.csv", index=False)
test_df.to_csv("data_preparation_accidents/test_datasets/traffic_accidents_test_3.csv", index=False)

print("Train/Test files created:")
print(f"Train: {len(train_df)} records")
print(f"Test:  {len(test_df)} records")

def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval, trnX, trnY, tstX, tstY  # Return data for confusion matrix


target = "crash_type"
file_tag = "crash"
train: DataFrame = read_csv("data_preparation_accidents/train_datasets/traffic_accidents_train_3.csv")
test: DataFrame = read_csv("data_preparation_accidents/test_datasets/traffic_accidents_test_3.csv")

# Evaluate metrics
figure()
eval, trnX, trnY, tstX, tstY = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"data_preparation_accidents/preparation_images/{file_tag}_eval_3.png")
show()

# --- Confusion matrices using scikit-learn ---
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay

# Train NB & KNN
nb_model = GaussianNB()
nb_model.fit(trnX, trnY)
pred_NB = nb_model.predict(tstX)

knn_model = KNeighborsClassifier()
knn_model.fit(trnX, trnY)
pred_KNN = knn_model.predict(tstX)

# Function to plot confusion matrix with percentages
def plot_conf_matrix(y_true, y_pred, title, save_path):
    min_label = min(y_true)
    y_true = y_true - min_label
    y_pred = y_pred - min_label
    cm = sk_confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, None] * 100  # row-wise percentages
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=sorted(set(y_true)))
    plt.fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".1f")
    ax.grid(False)     
    plt.title(title) 
    plt.tight_layout()
    savefig(save_path)
    show()

# Plot and save confusion matrices
plot_conf_matrix(tstY, pred_NB, "NB Confusion Matrix (%)", f"data_preparation_accidents/preparation_images/{file_tag}_NB_confusion_3.png")
plot_conf_matrix(tstY, pred_KNN, "KNN Confusion Matrix (%)", f"data_preparation_accidents/preparation_images/{file_tag}_KNN_confusion_3.png")