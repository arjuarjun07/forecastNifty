import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import Chronos2Pipeline

print("Loading data...")
# Load the data, telling pandas to ignore commas in numbers
df = pd.read_csv(
    "nifty_5min.csv", 
    thousands=',',
    parse_dates=['time'], 
    date_format="%d %b %y %I:%M %p" 
)

# Nifty spot doesn't have volume, so we drop it
df = df.drop(columns=["Volume"], errors="ignore")
# Ensure all prices are numeric
for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

# Create a continuous fake timestamp to bypass Pandas frequency inference errors (due to trading halts)
df["fake_time"] = pd.date_range("2024-01-01", periods=len(df), freq="5min")

# Format for Chronos
context_df = df.copy()
context_df["id"] = "NIFTY_5MIN"

# Keep only the last 2048 rows (max context)
context_df = context_df.tail(2048)

print("Loading Chronos-2 model...")
# amazon/chronos-2 is the best general model (state-of-the-art)
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2", 
    device_map="cpu" 
)

print("Generating multivariate forecast for the next 75 candles (1 trading day)...")
pred_length = 75

# Predict all 4 columns together
pred_df = pipeline.predict_df(
    context_df,
    prediction_length=pred_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="fake_time",
    target=["open", "high", "low", "close"],
)

# Extract just the Close predictions for plotting
pred_close = pred_df[pred_df["target_name"] == "close"]

print("Plotting results...")
# Show last few rows for context in the plot
ts_context = context_df["close"].tail(300).values
context_x = list(range(len(ts_context)))
forecast_x = list(range(len(ts_context), len(ts_context) + pred_length))

plt.figure(figsize=(14, 6))
plt.plot(context_x, ts_context, label="Historical Nifty 50 Close (5-min)", color="blue")
plt.plot(forecast_x, pred_close["predictions"].values, label="Median Forecast", color="purple")

# Plot prediction interval
plt.fill_between(
    forecast_x,
    pred_close["0.1"].values,
    pred_close["0.9"].values,
    alpha=0.3,
    label="80% Prediction Interval",
    color="purple"
)

plt.title("Nifty 50 Next-Day Multivariate Forecast (using Open/High/Low/Close)")
plt.xlabel("Steps (5-minute intervals)")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot instead of showing it
plt.savefig("forecast_output.png", bbox_inches='tight')
print("Plot saved to forecast_output.png")


