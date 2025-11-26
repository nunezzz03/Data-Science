from numpy import ndarray
from pandas import DataFrame, read_csv
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN
from pandas import read_csv
from sklearn.model_selection import train_test_split
import pandas as pd
from dslabs_functions import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = read_csv("data/raw/traffic_accidents.csv")
print(f"Original File: {len(df)} records")
df = df[(df != "unknown").all(axis=1)]

# Remove target-leakage columns
leak_cols = [
    "first_crash_type", "damage", "most_severe_injury", "injuries_total",
    "injuries_fatal", "injuries_incapacitating", "injuries_non_incapacitating",
    "injuries_reported_not_evident", "injuries_no_indication", "num_units", "prim_contributory_cause"
]
df = df.drop(columns=leak_cols, errors="ignore")

# Keep numeric + binary columns
numeric_df = df.select_dtypes(include=["number"])
binary_cols = [col for col in df.select_dtypes(include=["object"]).columns if df[col].nunique()==2]
binary_df = df[binary_cols].apply(lambda c: c.astype("category").cat.codes)

# Combine
df = pd.concat([numeric_df, binary_df], axis=1)

# Split 70/30
train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)

# Save
train_df.to_csv("data_preparation_accidents/train_datasets/traffic_accidents_train_1.csv", index=False)
test_df.to_csv("data_preparation_accidents/test_datasets/traffic_accidents_test_1.csv", index=False)

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
train: DataFrame = read_csv("data_preparation_accidents/train_datasets/traffic_accidents_train_1.csv")
test: DataFrame = read_csv("data_preparation_accidents/test_datasets/traffic_accidents_test_1.csv")

# Evaluate metrics
figure()
eval, trnX, trnY, tstX, tstY = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"data_preparation_accidents/preparation_images/{file_tag}_eval_1.png")
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
plot_conf_matrix(tstY, pred_NB, "NB Confusion Matrix (%)", f"data_preparation_accidents/preparation_images/{file_tag}_NB_confusion_1.png")
plot_conf_matrix(tstY, pred_KNN, "KNN Confusion Matrix (%)", f"data_preparation_accidents/preparation_images/{file_tag}_KNN_confusion_1.png")