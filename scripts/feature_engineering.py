import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

IN_PATH  = "../output/cleaning_output/cleaned.parquet"
OUT_DIR  = "../output/features_output"
os.makedirs(OUT_DIR, exist_ok=True)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)


# LOAD

df = pd.read_parquet(IN_PATH)
log(f"  Input shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
log(f"  Columns: {list(df.columns)}")


# 1. DROP CORRELATED / REDUNDANT COLUMNS
log("\n-- 1. Drop correlated / redundant columns --")

DROP_COLS = ["bcf", "sales_tax", "driver_pay", "trip_time"]
actual = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=actual, inplace=True)
log(f"  Dropped: {actual}")
log(f"  Remaining: {list(df.columns)}")


# 2. LOG TRANSFORMS (right-skewed continuous features)
log("\n-- 2. Log Transforms (log1p) --")

LOG_COLS = ["trip_miles", "base_passenger_fare", "trip_duration_min"]

fig, axes = plt.subplots(len(LOG_COLS), 2, figsize=(12, len(LOG_COLS) * 3.5))

sample = df.sample(min(300_000, len(df)), random_state=42)

for i, col in enumerate(LOG_COLS):
    if col not in df.columns:
        log(f"  SKIP {col} — not in dataset")
        continue

    log_col = f"log_{col}"
    df[log_col] = np.log1p(df[col])

    before_skew = sample[col].skew()
    after_skew  = np.log1p(sample[col]).skew()
    log(f"  {col:<30} skew before={before_skew:>6.2f}  "
        f"after={after_skew:>6.2f}  -> {log_col}")

    # Before plot
    axes[i, 0].hist(sample[col], bins=60, color="#FF5722",
                    alpha=0.75, edgecolor="none")
    axes[i, 0].set_title(f"{col} (raw)  skew={before_skew:.2f}",
                          fontsize=9, fontweight="bold")
    axes[i, 0].set_ylabel("Count")
    axes[i, 0].grid(True, alpha=0.2)

    # After plot
    axes[i, 1].hist(np.log1p(sample[col]), bins=60, color="#4CAF50",
                    alpha=0.75, edgecolor="none")
    axes[i, 1].set_title(f"{log_col}  skew={after_skew:.2f}",
                          fontsize=9, fontweight="bold")
    axes[i, 1].set_ylabel("Count")
    axes[i, 1].grid(True, alpha=0.2)

plt.suptitle("Log1p Transformation: Before vs After",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fe_01_log_transforms.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"\n  -> Saved fe_01_log_transforms.png")

# Drop originals — log versions replace them
df.drop(columns=[c for c in LOG_COLS if c in df.columns], inplace=True)
log(f"  Original columns dropped, log versions retained.")


# 3. ZERO-INFLATED COLUMNS -> BINARY FLAGS
log("\n-- 3. Zero-inflated columns -> binary flags --")

df["has_tip"]   = (df["tips"]  > 0).astype(int)
df["has_tolls"] = (df["tolls"] > 0).astype(int)

log(f"  has_tip   : {df['has_tip'].mean()*100:.1f}% of trips have a tip")
log(f"  has_tolls : {df['has_tolls'].mean()*100:.1f}% of trips have tolls")

df.drop(columns=["tips", "tolls"], inplace=True)
log(f"  Original tips and tolls columns dropped.")


# 4. DISCRETE COLUMNS -> BINARY FLAGS
log("\n-- 4. Discrete columns -> binary flags --")

df["is_congestion_zone"] = (df["congestion_surcharge"] > 0).astype(int)
df["is_airport_trip"]= (df["airport_fee"] > 0).astype(int)

log(f"  is_congestion_zone : {df['is_congestion_zone'].mean()*100:.1f}% of trips")
log(f"  is_airport_trip : {df['is_airport_trip'].mean()*100:.1f}% of trips")

df.drop(columns=["congestion_surcharge", "airport_fee"], inplace=True)


# 5. CATEGORICAL ENCODING
log("\n-- 5. Categorical encoding --")

# Operator: Uber=1, Lyft=0
df["is_uber"] = (df["hvfhs_license_num"] == "HV0003").astype(int)
log(f"  is_uber   : {df['is_uber'].mean()*100:.1f}% Uber, "
    f"{(1-df['is_uber'].mean())*100:.1f}% Lyft")
df.drop(columns=["hvfhs_license_num"], inplace=True)

# Shared match flag
df["is_shared"] = (df["shared_match_flag"] == "Y").astype(int)
log(f"  is_shared : {df['is_shared'].mean()*100:.3f}% shared trips")
df.drop(columns=["shared_match_flag"], inplace=True)

# WAV (wheelchair accessible)
df["is_wav"] = (df["wav_match_flag"] == "Y").astype(int)
log(f"  is_wav    : {df['is_wav'].mean()*100:.2f}% WAV trips")
df.drop(columns=["wav_match_flag"], inplace=True)


# 6. TEMPORAL FEATURE ENGINEERING
log("\n-- 6. Temporal features from pickup_datetime --")

df["hour_of_day"]     = df["pickup_datetime"].dt.hour
df["day_of_week"]     = df["pickup_datetime"].dt.dayofweek   # 0=Mon, 6=Sun
df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
df["is_peak_morning"] = df["hour_of_day"].between(7, 9).astype(int)
df["is_peak_evening"] = df["hour_of_day"].between(17, 20).astype(int)

log(f"  hour_of_day     : 0-23 (mean={df['hour_of_day'].mean():.1f})")
log(f"  day_of_week     : 0-6  (mean={df['day_of_week'].mean():.1f})")
log(f"  is_weekend      : {df['is_weekend'].mean()*100:.1f}% weekend trips")
log(f"  is_peak_morning : {df['is_peak_morning'].mean()*100:.1f}% morning peak")
log(f"  is_peak_evening : {df['is_peak_evening'].mean()*100:.1f}% evening peak")

# Drop datetime columns — no longer needed
df.drop(columns=["pickup_datetime", "dropoff_datetime"], inplace=True)


# 7. FINAL FEATURE SET
log("\n-- 7. Final Feature Set --")


FINAL_FEATURES = [
    # Continuous (log-transformed) — core trip characteristics
    "log_trip_miles",
    "log_base_passenger_fare",
    "log_trip_duration_min",
    # Strong binary separators
    "is_airport_trip",
    "is_congestion_zone",
    "has_tip",
    # Temporal demand signal
    "hour_of_day",
]

missing = [f for f in FINAL_FEATURES if f not in df.columns]
if missing:
    log(f"  WARNING — missing features: {missing}")

df = df[FINAL_FEATURES]

log(f"\n  Final shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
log(f"  Features: {FINAL_FEATURES}")
log(f"\n  Descriptive stats:")
log(df.describe().round(3).to_string())


# 8. VISUALISE FINAL FEATURES
n = len(FINAL_FEATURES)
cols_n = 3
rows_n = (n + cols_n - 1) // cols_n

sample2 = df.sample(min(300_000, len(df)), random_state=42)

fig, axes = plt.subplots(rows_n, cols_n, figsize=(15, rows_n * 3))
axes = axes.flatten()

for i, col in enumerate(FINAL_FEATURES):
    unique_vals = df[col].nunique()
    if unique_vals <= 10:
        # Bar chart for binary / low-cardinality
        vc = sample2[col].value_counts().sort_index()
        axes[i].bar(vc.index.astype(str), vc.values,
                    color="#9C27B0", alpha=0.8)
    else:
        # Histogram for continuous
        axes[i].hist(sample2[col], bins=50, color="#2196F3",
                     alpha=0.78, edgecolor="none")
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].grid(True, alpha=0.2)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Final Feature Set (post-engineering, pre-scaling)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fe_02_encoded_features.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"\n  -> Saved fe_02_encoded_features.png")


# 9. SAVE
out_path = f"{OUT_DIR}/features.parquet"
df.to_parquet(out_path, index=False)
log(f"  -> Saved features.parquet  ({os.path.getsize(out_path)/1e6:.1f} MB)")

with open(f"{OUT_DIR}/feature_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"  -> Saved feature_report.txt")
