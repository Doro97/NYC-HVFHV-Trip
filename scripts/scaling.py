import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")

IN_PATH  = "../output/features_output/features.parquet"
OUT_DIR  = "../output/scaling_output"
SAMPLE_N = 300_000
SEED     = 42
os.makedirs(OUT_DIR, exist_ok=True)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)


# LOAD
df = pd.read_parquet(IN_PATH)
log(f"  Full dataset : {df.shape[0]:,} rows x {df.shape[1]} columns")

# Sample for modelling
df_sample = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
log(f"  Sample       : {len(df_sample):,} rows (seed={SEED})")


# 1. DEFINE FEATURE GROUPS
log("\n-- 1. Feature Groups --")

STANDARD_SCALE = [
    "log_trip_miles",
    "log_base_passenger_fare",
    "log_trip_duration_min",
]

MINMAX_SCALE = [
    "hour_of_day",
]

NO_SCALE = [
    "is_airport_trip",
    "is_congestion_zone",
    "has_tip",
]

log(f"  StandardScaler : {STANDARD_SCALE}")
log(f"  MinMaxScaler : {MINMAX_SCALE}")
log(f"  No scaling : {NO_SCALE}")

ALL_FEATURES = STANDARD_SCALE + MINMAX_SCALE + NO_SCALE
missing = [f for f in ALL_FEATURES if f not in df_sample.columns]
if missing:
    log(f"\n  WARNING: missing features: {missing}")


# 2. FIT AND APPLY SCALERS
log("\n-- 2. Fitting and Applying Scalers --")

scaled_df = df_sample[ALL_FEATURES].copy()

# StandardScaler on continuous log-transformed features
std_scaler = StandardScaler()
scaled_df[STANDARD_SCALE] = std_scaler.fit_transform(
    df_sample[STANDARD_SCALE]
)
log(f"\n  StandardScaler fitted on {STANDARD_SCALE}")
for col, mean, std in zip(STANDARD_SCALE,
                           std_scaler.mean_,
                           std_scaler.scale_):
    log(f"    {col:<35}  mean={mean:.4f}  std={std:.4f}")

# MinMaxScaler on ordinal temporal features
mm_scaler = MinMaxScaler()
scaled_df[MINMAX_SCALE] = mm_scaler.fit_transform(
    df_sample[MINMAX_SCALE]
)
log(f"\n  MinMaxScaler fitted on {MINMAX_SCALE}")
for col, mn, mx in zip(MINMAX_SCALE,
                        mm_scaler.data_min_,
                        mm_scaler.data_max_):
    log(f"    {col:<35}  min={mn:.1f}  max={mx:.1f}  -> scaled to [0, 1]")

log(f"\n  Binary features left unscaled: {NO_SCALE}")


# 3. VERIFY SCALING
log("\n-- 3. Verification --")
log(f"\n  Mean and std after scaling (should be ~0/1 for StandardScaler,")
log(f"  ~0.5/0.28 for MinMaxScaler on uniform data):\n")

stats = scaled_df[STANDARD_SCALE + MINMAX_SCALE].agg(["mean", "std"]).round(4)
log(stats.to_string())

# Check no NaNs introduced
nan_count = scaled_df.isnull().sum().sum()
log(f"\n  NaN values after scaling: {nan_count}  "
    f"{'OK' if nan_count == 0 else 'WARNING'}")


# 4. BEFORE / AFTER PLOTS (continuous features only)
fig, axes = plt.subplots(len(STANDARD_SCALE), 2,
                          figsize=(12, len(STANDARD_SCALE) * 3.5))

s = df_sample.sample(100_000, random_state=SEED)

for i, col in enumerate(STANDARD_SCALE):
    # Before scaling
    axes[i, 0].hist(s[col], bins=60, color="#FF9800",
                    alpha=0.78, edgecolor="none")
    axes[i, 0].set_title(f"{col}  (pre-scaling)",
                          fontsize=9, fontweight="bold")
    axes[i, 0].set_ylabel("Count")
    axes[i, 0].grid(True, alpha=0.2)

    # After scaling
    axes[i, 1].hist(scaled_df.loc[s.index, col], bins=60,
                    color="#2196F3", alpha=0.78, edgecolor="none")
    scaled_mean = scaled_df[col].mean()
    scaled_std  = scaled_df[col].std()
    axes[i, 1].set_title(
        f"scaled_{col}  mean={scaled_mean:.2f}  std={scaled_std:.2f}",
        fontsize=9, fontweight="bold"
    )
    axes[i, 1].set_ylabel("Count")
    axes[i, 1].grid(True, alpha=0.2)

plt.suptitle("Scaling: Before vs After (StandardScaler)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sc_01_before_after.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"\n  -> Saved sc_01_before_after.png")


# 5. FULL SCALED FEATURE DISTRIBUTION PLOT
n_feat = len(ALL_FEATURES)
cols_n = 3
rows_n = (n_feat + cols_n - 1) // cols_n

fig, axes = plt.subplots(rows_n, cols_n,
                          figsize=(15, rows_n * 3))
axes = axes.flatten()

s2 = scaled_df.sample(100_000, random_state=SEED)

for i, col in enumerate(ALL_FEATURES):
    unique_vals = scaled_df[col].nunique()
    if unique_vals <= 10:
        vc = s2[col].value_counts().sort_index()
        axes[i].bar(vc.index.astype(str), vc.values,
                    color="#9C27B0", alpha=0.8)
    else:
        axes[i].hist(s2[col], bins=50, color="#2196F3",
                     alpha=0.78, edgecolor="none")
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].grid(True, alpha=0.2)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("All Features After Scaling (300k sample)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sc_02_scaled_distributions.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"  -> Saved sc_02_scaled_distributions.png")


# 6. SAVE
# Save scaled sample (this is what modelling will use)
out_path = f"{OUT_DIR}/scaled.parquet"
scaled_df.to_parquet(out_path, index=False)
log(f"\n  -> Saved scaled.parquet  ({os.path.getsize(out_path)/1e6:.1f} MB)")
log(f"     Shape: {scaled_df.shape[0]:,} rows x {scaled_df.shape[1]} columns")

# Save scalers (needed to inverse-transform cluster centres later)
scalers = {"standard": std_scaler, "minmax": mm_scaler,
           "standard_cols": STANDARD_SCALE, "minmax_cols": MINMAX_SCALE}
with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scalers, f)
log(f"  -> Saved scaler.pkl")

with open(f"{OUT_DIR}/scaling_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"  -> Saved scaling_report.txt")
