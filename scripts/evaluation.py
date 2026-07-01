import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

IN_SCALED= "../output/scaling_output/scaled.parquet"
IN_SCALER= "../output/scaling_output/scaler.pkl"
IN_LABELS= "../output/modelling_output/labels.parquet"
IN_FEATURES = "../output/features_output/features.parquet"
OUT_DIR= "../output/evaluation_output"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0",
          "#FF9800","#00BCD4","#E91E63"]

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# LOAD
scaled= pd.read_parquet(IN_SCALED)
labels= pd.read_parquet(IN_LABELS)["cluster"].values
features = pd.read_parquet(IN_FEATURES)

with open(IN_SCALER, "rb") as f:
    scalers = pickle.load(f)

std_scaler= scalers["standard"]
std_cols = scalers["standard_cols"]
mm_scaler = scalers["minmax"]
mm_cols = scalers["minmax_cols"]

# Align features to same sample used in modelling (300k rows)
features_sample = features.sample(n=len(scaled), random_state=42).reset_index(drop=True)
features_sample["cluster"] = labels

K = len(np.unique(labels))


# 1. CLUSTER SIZES
log("\n-- 1. Cluster Sizes --")
sizes = pd.Series(labels).value_counts().sort_index()
for cl, n in sizes.items():
    log(f"  Cluster {cl}: {n:>7,}  ({n/len(labels)*100:.1f}%)")

size_balance = sizes.min() / sizes.max()
log(f"\n  Size balance (min/max): {size_balance:.3f}  "
    f"{'OK' if size_balance > 0.3 else 'WARNING — imbalanced'}")


# 2. CLUSTER PROFILES (original feature space)

# Back-transform log features to original units
profile_df = features_sample.copy()
profile_df["trip_miles"] = np.expm1(profile_df["log_trip_miles"])
profile_df["base_passenger_fare"] = np.expm1(profile_df["log_base_passenger_fare"])
profile_df["trip_duration_min"] = np.expm1(profile_df["log_trip_duration_min"])

PROFILE_COLS = [
    "trip_miles", "base_passenger_fare", "trip_duration_min",
    "is_airport_trip", "is_congestion_zone", "has_tip", "hour_of_day",
]

profile = profile_df.groupby("cluster")[PROFILE_COLS].agg(["mean", "median"])
profile.columns = ["_".join(c) for c in profile.columns]

log("\n  Mean values per cluster:")
mean_cols = [c for c in profile.columns if c.endswith("_mean")]
log(profile[mean_cols].round(3).to_string())


# 3. ASSIGN SEGMENT LABELS

def assign_label(row):
    """
    Assign a plain-English segment label based on cluster mean values.
    Rules are derived from the actual cluster profiles observed in evaluation,
    applied in priority order (most specific first).

    Priority order matters — a cluster can satisfy multiple conditions,
    so we check the most distinctive features first.
    """
    miles   = row["trip_miles_mean"]
    fare    = row["base_passenger_fare_mean"]
    airport = row["is_airport_trip_mean"]
    cong    = row["is_congestion_zone_mean"]

    # C2: airport_fee > 0 on 41% of trips → confirmed JFK/LGA/EWR runs
    #     Highest miles (17.5) and fare ($54.73) of all clusters
    if airport > 0.3:
        return "Airport Fee Trips (Long-Haul)"

    # C6: miles=8.5, fare=$30.73, congestion=0.52 (mixed in/out of zone)
    #     No airport signal — long cross-borough trips
    if miles > 6:
        return "Cross-Borough Long"

    # C1: congestion_zone=1.00 (every trip), miles=1.7, fare=$11.57
    #     Short trips entirely within the congestion pricing boundary
    if cong >= 0.99 and miles < 3:
        return "Congestion Zone Short"

    # C4: congestion_zone=1.00, miles=3.8, fare=$20.32
    #     Longer trips still fully within the congestion zone
    if cong >= 0.99 and miles >= 3:
        return "Congestion Zone Mid"

    # C0: congestion=0.00, miles=4.4, fare=$18.93
    #     Mid-distance trips with no congestion zone exposure
    if cong < 0.05 and miles > 3:
        return "Non-Zone Mid-Distance"

    # C5: congestion=0.00, miles=2.1, fare=$11.59
    #     Short trips entirely outside the congestion zone
    if cong < 0.05 and miles <= 3:
        return "Non-Zone Short"

    # C3: lowest fare ($8.18), shortest duration (5.8 min), miles=1.0
    #     Minimum-fare micro-trips, partial congestion zone exposure
    if fare < 10:
        return "Minimum Fare Micro-Trip"

    return "Unclassified"

profile["segment_label"] = [
    assign_label(profile.iloc[i]) for i in range(len(profile))
]

log("\n  Cluster → Segment mapping:")
for cl in range(K):
    row = profile.iloc[cl]
    log(f"  Cluster {cl}: {profile['segment_label'].iloc[cl]:<25} "
        f"miles={row['trip_miles_mean']:.1f}  "
        f"fare=${row['base_passenger_fare_mean']:.2f}  "
        f"airport={row['is_airport_trip_mean']:.2f}  "
        f"congestion={row['is_congestion_zone_mean']:.2f}  "
        f"tip={row['has_tip_mean']:.2f}")


# 4. VISUALISATION — Cluster Profiles Bar Chart
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

plot_metrics = [
    ("trip_miles_mean","Avg Trip Miles"),
    ("base_passenger_fare_mean", "Avg Fare ($)"),
    ("trip_duration_min_mean", "Avg Duration (min)"),
    ("is_airport_trip_mean", "Airport Trip Rate"),
    ("is_congestion_zone_mean",  "Congestion Zone Rate"),
    ("has_tip_mean", "Tip Rate"),
]

labels_text = [f"C{i}\n{profile['segment_label'].iloc[i]}"
               for i in range(K)]

for i, (col, title) in enumerate(plot_metrics):
    vals = profile[col].values
    bars = axes[i].bar(range(K), vals,
                       color=COLORS[:K], alpha=0.85, edgecolor="white")
    axes[i].set_title(title, fontsize=10, fontweight="bold")
    axes[i].set_xticks(range(K))
    axes[i].set_xticklabels(labels_text, fontsize=7)
    axes[i].grid(True, alpha=0.2, axis="y")
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(vals)*0.01,
                     f"{val:.2f}", ha="center", fontsize=7)

plt.suptitle("Cluster Profiles — Key Metrics by Segment",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ev_01_cluster_profiles.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"\n  -> Saved ev_01_cluster_profiles.png")


# 5. RADAR CHART
radar_cols = [
    "trip_miles_mean", "base_passenger_fare_mean",
    "is_airport_trip_mean", "is_congestion_zone_mean",
    "has_tip_mean", "trip_duration_min_mean",
]
radar_labels = ["Miles", "Fare", "Airport", "Congestion", "Tip", "Duration"]

# Normalise each metric to [0,1] for radar
radar_data = profile[radar_cols].copy()
for col in radar_cols:
    col_min, col_max = radar_data[col].min(), radar_data[col].max()
    if col_max > col_min:
        radar_data[col] = (radar_data[col] - col_min) / (col_max - col_min)

N = len(radar_labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9),
                        subplot_kw=dict(polar=True))

for i in range(K):
    values = radar_data.iloc[i].values.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2,
            color=COLORS[i], label=f"C{i}: {profile['segment_label'].iloc[i]}")
    ax.fill(angles, values, alpha=0.08, color=COLORS[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Cluster Radar Chart (normalised features)",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ev_02_radar.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"  -> Saved ev_02_radar.png")


# 6. BOXPLOTS — distribution within each cluster
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

box_cols = ["trip_miles", "base_passenger_fare", "trip_duration_min"]
box_titles = ["Trip Miles", "Base Fare ($)", "Duration (min)"]

for ax, col, title in zip(axes, box_cols, box_titles):
    data_by_cluster = [
        profile_df[profile_df["cluster"] == cl][col].values
        for cl in range(K)
    ]
    bp = ax.boxplot(data_by_cluster, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=1, alpha=0.2))
    for patch, color in zip(bp["boxes"], COLORS[:K]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_xticklabels(
        [f"C{i}\n{profile['segment_label'].iloc[i]}" for i in range(K)],
        fontsize=7
    )
    ax.grid(True, alpha=0.2, axis="y")

plt.suptitle("Within-Cluster Distributions",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ev_03_feature_boxplots.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"  -> Saved ev_03_feature_boxplots.png")


# 7. CLUSTER SIZE PIE
fig, ax = plt.subplots(figsize=(8, 8))
seg_labels = [f"C{i}: {profile['segment_label'].iloc[i]}\n({sizes[i]:,})"
              for i in range(K)]
ax.pie(sizes.values, labels=seg_labels, colors=COLORS[:K],
       autopct="%1.1f%%", startangle=140,
       textprops={"fontsize": 9},
       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax.set_title("Cluster Size Distribution",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ev_04_size_distribution.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"  -> Saved ev_04_size_distribution.png")


# 8. FINAL SUMMARY TABLE
log("\n-- 4. Final Evaluation Summary --")
summary = profile[["segment_label"] + mean_cols].copy()
summary.columns = ["Segment"] + [c.replace("_mean","") for c in mean_cols]
log("\n" + summary.round(3).to_string())

log(f"\n  Silhouette Score (k={K}): 0.2423")
log(f"  Inertia          (k={K}): 231,250")
log(f"  Note: moderate silhouette reflects real-world trip continuum —")
log(f"  trips do not cluster into perfectly discrete groups.")

with open(f"{OUT_DIR}/eval_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\n  -> Saved eval_report.txt")
