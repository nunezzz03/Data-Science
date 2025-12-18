# =========================
# Imports
# =========================
from copy import deepcopy
from pathlib import Path
import torch
from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import subplots, savefig

from dslabs_functions import (
    FORECAST_MEASURES,
    DELTA_IMPROVE,
    HEIGHT,
    plot_multiline_chart,
    plot_forecasting_eval,
    plot_forecasting_series,
)
Path("images").mkdir(exist_ok=True)

# =========================
# Dataset preparation
# =========================
def prepare_dataset_for_lstm(series, seq_length: int = 4):
    setX, setY = [], []
    for i in range(len(series) - seq_length):
        past = series[i : i + seq_length]
        future = series[i + 1 : i + seq_length + 1]
        setX.append(past)
        setY.append(future)
    if len(setX) == 0:
        return tensor([]), tensor([])
    return tensor(setX), tensor(setY)

# =========================
# LSTM Model
# =========================
class DS_LSTM(Module):
    def __init__(self, train, input_size: int = 1, hidden_size: int = 8,
                 num_layers: int = 1, length: int = 2, lr: float = 1e-3):
        super().__init__()

        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                         num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_size, 1)
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = MSELoss()

        trnX, trnY = prepare_dataset_for_lstm(train, seq_length=length)
        batch_size = max(1, len(trnX) // 10)
        self.loader = DataLoader(TensorDataset(trnX, trnY),
                                 shuffle=True, batch_size=batch_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def fit(self):
        self.train()
        epoch_loss = 0.0
        for batchX, batchY in self.loader:
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / max(len(self.loader), 1)

    def predict(self, X):
        self.eval()
        if len(X) == 0:
            return tensor([])
        with no_grad():
            y_pred = self(X)
        return y_pred[:, -1, :]

# =========================
# Load data
# =========================
filename = Path(__file__).parent / "economic_usa_smoothed.csv"
file_tag = "ECONOMICS"
target = "Inflation Rate (%)"
timecol = None  # default index

measure = "MAE"

data: DataFrame = read_csv(filename)
series = data[[target]].values.astype("float32")

train_size = int(len(series) * 0.90)
train, test = series[:train_size], series[train_size:]

# =========================
# Hyperparameter study
# =========================
def lstm_study(train, test, nr_episodes: int = 1500, measure: str = "MAE"):
    sequence_size = [2]  # tiny dataset only allows 2
    nr_hidden_units = [8, 16, 32]

    step = max(nr_episodes // 10, 1)
    episodes = [1] + list(range(step, nr_episodes + 1, step))

    flag = measure in ("R2", "MAPE")
    best_model = None
    best_performance = -1e9
    best_params = {"name": "LSTM", "metric": measure, "params": ()}

    # Safe plotting
    _, axs = subplots(1, len(sequence_size), figsize=(len(sequence_size) * HEIGHT, HEIGHT))
    if len(sequence_size) == 1:
        axs = [axs]

    for i, length in enumerate(sequence_size):
        # skip invalid sequences
        if len(test) <= length:
            print(f"Skipping seq_length={length} (test set too small)")
            continue

        print(f"\n=== Sequence length = {length} ===")
        tstX, _ = prepare_dataset_for_lstm(test, seq_length=length)

        values = {}
        for hidden in nr_hidden_units:
            print(f"\n-- Hidden units = {hidden} --")
            yvalues = []

            model = DS_LSTM(train, hidden_size=hidden, length=length)

            for n in range(1, nr_episodes + 1):
                loss = model.fit()

                if n % step == 0 or n == 1:
                    prd_tst = model.predict(tstX)
                    if len(prd_tst) == 0:
                        eval_value = float("nan")
                    else:
                        eval_value = FORECAST_MEASURES[measure](test[length:], prd_tst)

                    print(f"seq={length} hidden={hidden} epochs={n} "
                          f"loss={loss:.5f} {measure}={eval_value:.4f}")

                    if not eval_value != eval_value:  # check for NaN
                        if eval_value > best_performance and abs(eval_value - best_performance) > DELTA_IMPROVE:
                            best_performance = eval_value
                            best_params["params"] = (length, hidden, n)
                            best_model = deepcopy(model)

                    yvalues.append(eval_value)

            values[hidden] = yvalues

        plot_multiline_chart(
            episodes,
            values,
            ax=axs[i],
            title=f"LSTM seq length={length} ({measure})",
            xlabel="nr epochs",
            ylabel=measure,
            percentage=flag,
        )

    print(f"\nBEST LSTM → length={best_params['params'][0]} "
          f"hidden={best_params['params'][1]} "
          f"epochs={best_params['params'][2]} "
          f"{measure}={best_performance:.4f}")

    savefig(f"images/{file_tag}_lstms_{measure}_study.png")
    return best_model, best_params

# =========================
# Run study
# =========================
best_model, best_params = lstm_study(train, test, nr_episodes=1500, measure=measure)

# =========================
# Final evaluation
# =========================
best_length, best_hidden, best_epochs = best_params["params"]

trnX, _ = prepare_dataset_for_lstm(train, seq_length=best_length)
tstX, _ = prepare_dataset_for_lstm(test, seq_length=best_length)

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plot_forecasting_eval(
    train[best_length:],
    test[best_length:],
    prd_trn,
    prd_tst,
    title=f"{file_tag} - LSTM (length={best_length}, hidden={best_hidden}, epochs={best_epochs})",
)
savefig(f"images/{file_tag}_lstms_{measure}_eval.png")

# =========================
# Forecast series plot
# =========================
series_df = DataFrame(series, columns=[target])
train_df = series_df[:train_size]
test_df = series_df[train_size:]

pred_series = Series(
    prd_tst.numpy().ravel() if len(prd_tst) > 0 else [],
    index=test_df.index[best_length:] if len(test_df) > best_length else [],
)

plot_forecasting_series(
    train_df[best_length:],
    test_df[best_length:],
    pred_series,
    title=f"{file_tag} - LSTM Forecast",
    xlabel="time",
    ylabel=target,
)
savefig(f"images/{file_tag}_lstms_{measure}_forecast.png")
