
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings("ignore")

# Config 
PARQUET_FILE = "../data/fhvhv_tripdata_2022-01.parquet"
OUT_DIR = "../eda_output"
os.makedirs(OUT_DIR, exist_ok=True)

# All numeric columns in the TLC HVFHV schema
NUMERIC_COLS = [
    "trip_miles",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
]

# Categorical columns worth inspecting
CATEGORICAL_COLS = [
    "hvfhs_license_num",   # Uber vs Lyft
    "dispatching_base_num",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]



# LOAD

print("Loading data...")
df = pd.read_parquet(PARQUET_FILE)
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")



# 1. DTYPES, NULLS, UNIQUE COUNTS


overview = pd.DataFrame({
    "dtype":df.dtypes,
    "nulls": df.isnull().sum(),
    "null_pct": (df.isnull().mean() * 100).round(2),
    "n_unique": df.nunique(),
    "sample":[str(df[c].dropna().iloc[0]) if df[c].notna().any() else "ALL NULL"
                 for c in df.columns],
})
print(overview.to_string())

with open(f"{OUT_DIR}/eda_01_dtypes.txt", "w") as f:
    f.write(overview.to_string())
print(f"\n  → Saved eda_01_dtypes.txt")



# 2. NUMERIC DISTRIBUTIONS (histograms)



# Use a sample for speed
sample = df.sample(min(500_000, len(df)), random_state=42)
available_num = [c for c in NUMERIC_COLS if c in df.columns]

# Also derive trip_duration_min here for inspection
sample["trip_duration_min"] = (
    (sample["dropoff_datetime"] - sample["pickup_datetime"])
    .dt.total_seconds().div(60)
)
available_num_plot = available_num + ["trip_duration_min"]

n = len(available_num_plot)
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(available_num_plot):
    data = sample[col].dropna()
    # Cap at 99th pct for readable histogram
    cap = data.quantile(0.99)
    data_capped = data[data <= cap]

    axes[i].hist(data_capped, bins=60, color="#2196F3", alpha=0.75, edgecolor="none")
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].grid(True, alpha=0.2)

    median = data.median()
    axes[i].axvline(median, color="red", linestyle="--", linewidth=1.2,
                    label=f"median={median:.2f}")
    axes[i].legend(fontsize=7)

    pct_negative = (data < 0).mean() * 100
    pct_zero = (data == 0).mean() * 100
    print(f"  {col:<30}  median={median:>8.2f}  "
          f"negative={pct_negative:>5.1f}%  zero={pct_zero:>5.1f}%  "
          f"99th={data.quantile(0.99):>8.2f}  max={data.max():>10.2f}")

# Hide unused axes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Numeric Feature Distributions (capped at 99th pct)", 
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_02_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → Saved eda_02_distributions.png")



# 3. SKEWNESS & KURTOSIS


skew_df = sample[available_num_plot].agg(["skew", "kurt"]).T
skew_df.columns = ["skewness", "kurtosis"]
skew_df["skew_flag"] = skew_df["skewness"].abs().apply(
    lambda x: "HIGH — log transform recommended" if x > 2 else
              ("MODERATE" if x > 1 else "OK")
)
print(skew_df.to_string())

with open(f"{OUT_DIR}/eda_03_skewness.txt", "w") as f:
    f.write(skew_df.to_string())
print(f"\n  → Saved eda_03_skewness.txt")



# 4. CORRELATION HEATMAP
corr = sample[available_num_plot].corr()
print(corr.round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5,
                color="white" if abs(val) > 0.6 else "black")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_04_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → Saved eda_04_correlation.png")

# Flag high correlations
print("\n  Pairs with |r| > 0.80:")
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > 0.80:
            print(f"    {corr.columns[i]:<30} ↔  {corr.columns[j]:<30}  r={r:.3f}")



# 5. OUTLIER BOXPLOTS
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(available_num_plot):
    data = sample[col].dropna()
    # Show raw (uncapped) data so outlier extent is visible
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#90CAF9"),
                    medianprops=dict(color="red", linewidth=2),
                    flierprops=dict(marker=".", markersize=1, alpha=0.3))
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Value")
    axes[i].grid(True, alpha=0.2)

    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
    pct = n_outliers / len(data) * 100
    print(f"  {col:<30}  IQR outliers: {n_outliers:>7,} ({pct:.1f}%)")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Outlier Extent (raw values)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_05_outliers.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → Saved eda_05_outliers.png")



# 6. CATEGORICAL VALUE COUNTS
available_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
lines = []
for col in available_cat:
    vc = df[col].value_counts(dropna=False).head(10)
    block = f"\n{col}:\n{vc.to_string()}"
    print(block)
    lines.append(block)

with open(f"{OUT_DIR}/eda_06_categoricals.txt", "w") as f:
    f.write("\n".join(lines))
print(f"\n  → Saved eda_06_categoricals.txt")



# 7. TEMPORAL PATTERNS

sample["hour"]    = sample["pickup_datetime"].dt.hour
sample["weekday"] = sample["pickup_datetime"].dt.day_name()

WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Trips by hour
hourly = sample.groupby("hour").size()
ax1.bar(hourly.index, hourly.values, color="#2196F3", alpha=0.8)
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Number of Trips (sample)")
ax1.set_title("Trip Volume by Hour of Day", fontweight="bold")
ax1.set_xticks(range(0, 24))
ax1.grid(True, alpha=0.2, axis="y")

# Trips by weekday
daily = sample.groupby("weekday").size().reindex(WEEKDAY_ORDER)
ax2.bar(daily.index, daily.values, color="#FF5722", alpha=0.8)
ax2.set_xlabel("Day of Week")
ax2.set_ylabel("Number of Trips (sample)")
ax2.set_title("Trip Volume by Day of Week", fontweight="bold")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(True, alpha=0.2, axis="y")

plt.suptitle("Temporal Demand Patterns — January 2022", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_07_temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → Saved eda_07_temporal.png")

print("\n  Busiest hours:")
print(hourly.sort_values(ascending=False).head(5).to_string())
print("\n  Trips by weekday:")
print(daily.to_string())
