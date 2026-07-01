import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# Config
PARQUET_FILE = "../data/fhvhv_tripdata_2022-01.parquet"
OUT_DIR = "../output/cleaning_output"
os.makedirs(OUT_DIR, exist_ok=True)

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)



# LOAD

log("=" * 60)
log("STEP 2: DATA CLEANING")
log("=" * 60)
log(f"\nLoading {PARQUET_FILE} ...")
df = pd.read_parquet(PARQUET_FILE)
log(f"  Raw shape : {df.shape[0]:>12,} rows × {df.shape[1]} columns")



# 1. DROP IRRELEVANT / HIGH-NULL COLUMNS

log("\n── 1. Column Drops ──")

DROP_COLS = [
    "originating_base_num",   # 26.6% null
    "on_scene_datetime",      # 26.6% null
    "dispatching_base_num",   # redundant with hvfhs_license_num
    "request_datetime",       # use pickup_datetime instead
    "shared_request_flag",    # 100% 'N'
    "access_a_ride_flag",     # mostly blank
    "wav_request_flag",       # 99.9% 'N'
    "PULocationID",           # raw zone IDs not meaningful for K-Means
    "DOLocationID",           # same
]

actual_drops = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=actual_drops, inplace=True)
log(f"  Dropped {len(actual_drops)} columns: {actual_drops}")
log(f"  Remaining columns ({df.shape[1]}): {list(df.columns)}")



# 2. DERIVE TRIP DURATION

log("\n── 2. Derive trip_duration_min ──")
df["trip_duration_min"] = (
    (df["dropoff_datetime"] - df["pickup_datetime"])
    .dt.total_seconds()
    .div(60)
)
log(f"  trip_duration_min derived from pickup/dropoff timestamps.")
log(f"  Range: {df['trip_duration_min'].min():.2f} → "
    f"{df['trip_duration_min'].max():.2f} minutes")



# 3. DROP INVALID ROWS

log("\n── 3. Row-level Cleaning ──")

def drop_rows(df, mask, reason):
    n = mask.sum()
    df = df[~mask].copy()
    log(f"  Dropped {n:>8,} rows  ({n/len(df)*100:.2f}%)  — {reason}")
    return df

before = len(df)

# Negative or zero fares / pay (refunds, corrections)
df = drop_rows(df, df["base_passenger_fare"] < 0,
               "negative base_passenger_fare (refunds)")
df = drop_rows(df, df["driver_pay"] < 0,
               "negative driver_pay (refunds)")

# Impossible trip geometry
df = drop_rows(df, df["trip_miles"] <= 0,
               "trip_miles <= 0 (stationary / cancelled)")
df = drop_rows(df, df["trip_duration_min"] <= 0,
               "trip_duration_min <= 0 (impossible timestamps)")

# Suspiciously short trips (under 1 minute — likely data errors)
df = drop_rows(df, df["trip_duration_min"] < 1,
               "trip_duration_min < 1 min (likely data error)")

# Negative tips or tolls
df = drop_rows(df, df["tips"] < 0,   "negative tips")
df = drop_rows(df, df["tolls"] < 0,  "negative tolls")

# Exact duplicates (same pickup time, dropoff time, fare, miles)
dup_cols = ["pickup_datetime", "dropoff_datetime",
            "trip_miles", "base_passenger_fare"]
n_dups = df.duplicated(subset=dup_cols).sum()
df = df.drop_duplicates(subset=dup_cols)
log(f"  Dropped {n_dups:>8,} rows               — exact duplicates")

after = len(df)
log(f"\n  Total rows removed : {before - after:,}  "
    f"({(before - after) / before * 100:.2f}%)")
log(f"  Rows remaining     : {after:,}")



# 4. OUTLIER CAPPING

# Strategy depends on distribution type:
#
#   Continuous right-skewed (trip_miles, base_passenger_fare, driver_pay,
#   trip_duration_min): cap at 99.5th percentile.
#   IQR-based fencing is NOT used here because these distributions are
#   zero-inflated or heavily skewed — IQR×3 produces an upper fence that
#   is too low, creating an artificial spike at the cap boundary.
#
#   Zero-inflated (tips, tolls): DO NOT cap.
#   Median and Q3 are both 0, so IQR=0 and any fence = 0, which wipes
#   all non-zero values. These will be converted to binary flags in
#   feature engineering instead.

log("\n── 4. Outlier Capping ──")
log("  Continuous features : 99.5th percentile cap")
log("  Zero-inflated (tips, tolls) : NOT capped — will become binary flags\n")

# Continuous columns — 99.5th percentile cap
CONTINUOUS_CAP_COLS = [
    "trip_miles",
    "base_passenger_fare",
    "driver_pay",
    "trip_duration_min",
]

for col in CONTINUOUS_CAP_COLS:
    if col not in df.columns:
        continue
    upper = df[col].quantile(0.995)
    n_capped = (df[col] > upper).sum()
    df[col] = df[col].clip(upper=upper)
    log(f"  {col:<30}  99.5th pct cap={upper:>8.2f}  "
        f"capped {n_capped:>6,} rows ({n_capped/len(df)*100:.2f}%)")

# Zero-inflated — report distribution only, no capping
log("")
for col in ["tips", "tolls"]:
    if col not in df.columns:
        continue
    n_nonzero = (df[col] > 0).sum()
    pct = n_nonzero / len(df) * 100
    log(f"  {col:<30}  non-zero: {n_nonzero:>8,} ({pct:.2f}%)  — NOT capped")

log(f"\n  Shape after capping: {df.shape[0]:,} rows x {df.shape[1]} columns")



# 5. FINAL NULL CHECK

log("\n── 5. Final Null Check ──")
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls) == 0:
    log("  No nulls remaining. OK")
else:
    log(f"  Remaining nulls:\n{nulls.to_string()}")
    df.dropna(inplace=True)
    log(f"  Dropped remaining null rows. Shape: {df.shape}")



# 6. FINAL SHAPE & COLUMN LIST

log("\n── 6. Final Dataset ──")
log(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
log(f"  Columns: {list(df.columns)}")
log("\n  Descriptive stats (numeric):")
log(df.describe().round(2).to_string())



# 7. POST-CLEANING DISTRIBUTIONS

PLOT_COLS = ["trip_miles", "base_passenger_fare", "trip_duration_min",
             "tips", "tolls", "congestion_surcharge", "airport_fee", "driver_pay"]
available = [c for c in PLOT_COLS if c in df.columns]

sample = df.sample(min(300_000, len(df)), random_state=42)
n = len(available)
cols_n = 3
rows_n = (n + cols_n - 1) // cols_n

fig, axes = plt.subplots(rows_n, cols_n, figsize=(15, rows_n * 3.5))
axes = axes.flatten()

for i, col in enumerate(available):
    axes[i].hist(sample[col], bins=60, color="#4CAF50", alpha=0.78,
                 edgecolor="none")
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].grid(True, alpha=0.2)
    median = sample[col].median()
    axes[i].axvline(median, color="red", linestyle="--", linewidth=1.2,
                    label=f"median={median:.2f}")
    axes[i].legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Post-Cleaning Distributions (before log transform)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cleaning_03_post_distributions.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"\n  → Saved cleaning_03_post_distributions.png")



# 8. SAVE

out_path = f"{OUT_DIR}/cleaned.parquet"
df.to_parquet(out_path, index=False)
log(f"  → Saved cleaned.parquet  ({os.path.getsize(out_path)/1e6:.1f} MB)")

with open(f"{OUT_DIR}/cleaning_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines), )
log(f"  → Saved cleaning_report.txt")

