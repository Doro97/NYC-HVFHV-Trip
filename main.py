import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(
    description="NYC HVFHV K-Means Clustering Pipeline"
)
parser.add_argument("--skip-eda", action="store_true",
                    help="Skip Step 1 (EDA) for faster re-runs")
parser.add_argument("--steps", type=str, default=None,
                    help="Comma-separated steps to run, e.g. '2,3,4,5,6'")
parser.add_argument("--data", type=str,
                    default="../data/fhvhv_tripdata_2022-01.parquet",
                    help="Path to raw parquet file")
args = parser.parse_args()

if args.steps:
    run_steps = set(int(s) for s in args.steps.split(","))
else:
    run_steps = {1, 2, 3, 4, 5, 6}
    if args.skip_eda:
        run_steps.discard(1)

PARQUET_FILE = args.data


def banner(step, title):
    print(f"\n{'='*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'='*60}")


def check_file(path, step_name):
    if not os.path.exists(path):
        print(f"\n  ERROR: Required file not found: {path}")
        print(f"  Please run the preceding step before {step_name}.")
        sys.exit(1)


total_start = time.time()


# EDA
if 1 in run_steps:
    banner(1, "EXPLORATORY DATA ANALYSIS")
    check_file(PARQUET_FILE, "EDA")
    t = time.time()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os as _os
    _os.makedirs("eda_output", exist_ok=True)
    
    df = pd.read_parquet(PARQUET_FILE)

    NUMERIC_COLS = ["trip_miles","base_passenger_fare","tolls","bcf",
                    "sales_tax","congestion_surcharge","airport_fee",
                    "tips","driver_pay"]
    available = [c for c in NUMERIC_COLS if c in df.columns]
    sample = df.sample(min(500_000, len(df)), random_state=42)
    sample["trip_duration_min"] = (
        (sample["dropoff_datetime"] - sample["pickup_datetime"])
        .dt.total_seconds().div(60)
    )
    available_plot = available + ["trip_duration_min"]

    # Null rates
    print("\n  Null rates:")
    for col in available:
        rate = df[col].isnull().mean() * 100
        print(f"    {col:<30} {rate:.2f}%")

    # Skewness
    print("\n  Skewness:")
    for col in available_plot:
        sk = sample[col].skew()
        flag = " <- HIGH" if abs(sk) > 2 else ""
        print(f"    {col:<30} {sk:>7.2f}{flag}")

    # Correlation — flag high pairs
    corr = sample[available_plot].corr()
    print("\n  High correlations (|r| > 0.80):")
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i,j]
            if abs(r) > 0.80:
                print(f"    {corr.columns[i]} <-> {corr.columns[j]}  r={r:.3f}")

    # Temporal
    sample["hour"] = sample["pickup_datetime"].dt.hour
   


# CLEANING
if 2 in run_steps:
    banner(2, "DATA CLEANING")
    check_file(PARQUET_FILE, "Cleaning")
    t = time.time()

    import pandas as pd
    import numpy as np
    import os as _os
    _os.makedirs("cleaning_output", exist_ok=True)

    df = pd.read_parquet(PARQUET_FILE)

    # Drop columns
    DROP_COLS = ["originating_base_num","on_scene_datetime",
                 "dispatching_base_num","request_datetime",
                 "shared_request_flag","access_a_ride_flag",
                 "wav_request_flag","PULocationID","DOLocationID"]
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Derive duration
    df["trip_duration_min"] = (
        (df["dropoff_datetime"] - df["pickup_datetime"])
        .dt.total_seconds().div(60)
    )

    # Remove invalid rows
    before = len(df)
    df = df[df["base_passenger_fare"] >= 0]
    df = df[df["driver_pay"] >= 0]
    df = df[df["trip_miles"] > 0]
    df = df[df["trip_duration_min"] >= 1]
    df = df[df["tips"] >= 0]
    df = df[df["tolls"] >= 0]
    df.drop_duplicates(
        subset=["pickup_datetime","dropoff_datetime","trip_miles","base_passenger_fare"],
        inplace=True
    )

    # Cap continuous features at 99.5th percentile
    for col in ["trip_miles","base_passenger_fare","driver_pay","trip_duration_min"]:
        if col in df.columns:
            cap = df[col].quantile(0.995)
            df[col] = df[col].clip(upper=cap)

    df.to_parquet("cleaning_output/cleaned.parquet", index=False)



# FEATURE ENGINEERING

if 3 in run_steps:
    banner(3, "FEATURE ENGINEERING & ENCODING")
    check_file("cleaning_output/cleaned.parquet", "Feature Engineering")
    t = time.time()

    import pandas as pd
    import numpy as np
    import os as _os
    _os.makedirs("features_output", exist_ok=True)

    df = pd.read_parquet("cleaning_output/cleaned.parquet")

    # Drop correlated/redundant
    df.drop(columns=[c for c in ["bcf","sales_tax","driver_pay","trip_time"]
                     if c in df.columns], inplace=True)

    # Log1p transform (reduces skewness)
    for col in ["trip_miles","base_passenger_fare","trip_duration_min"]:
        df[f"log_{col}"] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)

    # Zero-inflated -> binary flags
    df["has_tip"] = (df["tips"]  > 0).astype(int)
    df["has_tolls"] = (df["tolls"] > 0).astype(int)
    df.drop(columns=["tips","tolls"], inplace=True)

    # Discrete -> binary flags
    df["is_congestion_zone"] = (df["congestion_surcharge"] > 0).astype(int)
    df["is_airport_trip"] = (df["airport_fee"] > 0).astype(int)
    df.drop(columns=["congestion_surcharge","airport_fee"], inplace=True)

    # Categorical encoding
    df["is_uber"] = (df["hvfhs_license_num"] == "HV0003").astype(int)
    df["is_shared"] = (df["shared_match_flag"] == "Y").astype(int)
    df["is_wav"] = (df["wav_match_flag"] == "Y").astype(int)
    df.drop(columns=["hvfhs_license_num","shared_match_flag","wav_match_flag"],
            inplace=True)

    # Temporal features
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_morning"] = df["hour_of_day"].between(7,9).astype(int)
    df["is_peak_evening"] = df["hour_of_day"].between(17,20).astype(int)
    df.drop(columns=["pickup_datetime","dropoff_datetime"], inplace=True)

    # Final feature selection (7 high-signal features for clustering)
    FINAL_FEATURES = [
        "log_trip_miles", "log_base_passenger_fare", "log_trip_duration_min",
        "is_airport_trip", "is_congestion_zone", "has_tip", "hour_of_day",
    ]
    df = df[FINAL_FEATURES]
    df.to_parquet("features_output/features.parquet", index=False)
   



# SCALING
if 4 in run_steps:
    banner(4, "FEATURE SCALING")
    check_file("features_output/features.parquet", "Scaling")
    t = time.time()

    import pandas as pd
    import pickle
    import os as _os
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    _os.makedirs("scaling_output", exist_ok=True)

    SAMPLE_N = 300_000
    df = pd.read_parquet("features_output/features.parquet")
    df_sample = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)

    STANDARD_SCALE = ["log_trip_miles","log_base_passenger_fare","log_trip_duration_min"]
    MINMAX_SCALE = ["hour_of_day"]
    NO_SCALE = ["is_airport_trip","is_congestion_zone","has_tip"]

    scaled_df = df_sample.copy()
    std_scaler = StandardScaler()
    scaled_df[STANDARD_SCALE] = std_scaler.fit_transform(df_sample[STANDARD_SCALE])

    mm_scaler = MinMaxScaler()
    scaled_df[MINMAX_SCALE] = mm_scaler.fit_transform(df_sample[MINMAX_SCALE])   

    scaled_df.to_parquet("scaling_output/scaled.parquet", index=False)
    scalers = {"standard": std_scaler, "minmax": mm_scaler,
               "standard_cols": STANDARD_SCALE, "minmax_cols": MINMAX_SCALE}
    with open("scaling_output/scaler.pkl","wb") as f:
        pickle.dump(scalers, f)
    



# MODELLING

if 5 in run_steps:
    banner(5, "K-MEANS MODELLING")
    check_file("scaling_output/scaled.parquet", "Modelling")
    t = time.time()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import os as _os
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    _os.makedirs("modelling_output", exist_ok=True)

    df = pd.read_parquet("scaling_output/scaled.parquet")
    X  = df.values

    # Elbow + silhouette sweep
  
    results = []
    prev = None
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=20_000, random_state=42)
        delta = f"{prev - km.inertia_:>10,.0f}" if prev else "         —"
        print(f"  {k:>4}  {km.inertia_:>12,.0f}  {sil:>12.4f}  {delta}")
        results.append({"k":k,"inertia":km.inertia_,"silhouette":sil})
        prev = km.inertia_

    # K selection: k=7 — local silhouette peak, visually distinct PCA clusters
    CHOSEN_K = 7
    

    final_model = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=20)
    final_labels = final_model.fit_predict(X)
    final_sil = silhouette_score(X, final_labels, sample_size=20_000, random_state=42)
    print(f" Final inertia   : {final_model.inertia_:,.0f}")
    print(f" Final silhouette: {final_sil:.4f}")

    unique, counts = np.unique(final_labels, return_counts=True)
    for cl, cnt in zip(unique, counts):
        print(f" Cluster {cl}: {cnt:>7,}  ({cnt/len(final_labels)*100:.1f}%)")

    # Save model and labels
    with open("modelling_output/model.pkl","wb") as f:
        pickle.dump({"model":final_model,"k":CHOSEN_K,"features":list(df.columns)}, f)
    pd.DataFrame({"cluster":final_labels}).to_parquet(
        "modelling_output/labels.parquet", index=False)

    # Elbow plot
    res = pd.DataFrame(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
    fig.suptitle("K Selection: Elbow Curve + Silhouette Score",
                 fontsize=13, fontweight="bold")
    ax1.plot(res["k"], res["inertia"], "bo-", lw=2, ms=8)
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Curve"); ax1.grid(True, alpha=0.3)
    ax2.plot(res["k"], res["silhouette"], "rs-", lw=2, ms=8)
    ax2.axvline(CHOSEN_K, color="green", linestyle="--", alpha=0.7,
                label=f"chosen k={CHOSEN_K}")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("modelling_output/mod_01_elbow.png", dpi=150, bbox_inches="tight")
    plt.close()

    # PCA 2D plot
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0",
              "#FF9800","#00BCD4","#E91E63"]
    idx = np.random.default_rng(42).choice(len(X_pca), 50_000, replace=False)
    fig, ax = plt.subplots(figsize=(10,7))
    for cl, cnt in zip(unique, counts):
        mask = final_labels[idx] == cl
        ax.scatter(X_pca[idx][mask,0], X_pca[idx][mask,1],
                   s=2, alpha=0.35, color=COLORS[cl],
                   label=f"Cluster {cl} (n={cnt:,})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"K-Means Clusters (k={CHOSEN_K}) — PCA 2D Projection",
                 fontsize=12, fontweight="bold")
    ax.legend(markerscale=6, fontsize=9); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("modelling_output/mod_02_clusters_2d.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# EVALUATION
if 6 in run_steps:
    banner(6, "CLUSTER EVALUATION & PROFILING")
    check_file("modelling_output/labels.parquet", "Evaluation")
    t = time.time()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import os as _os
    _os.makedirs("evaluation_output", exist_ok=True)

    scaled   = pd.read_parquet("scaling_output/scaled.parquet")
    labels   = pd.read_parquet("modelling_output/labels.parquet")["cluster"].values
    features = pd.read_parquet("features_output/features.parquet")
    features_sample = features.sample(n=len(scaled), random_state=42).reset_index(drop=True)
    features_sample["cluster"] = labels

    K = len(np.unique(labels))
    COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0",
              "#FF9800","#00BCD4","#E91E63"]

    # Back-transform log features
    profile_df = features_sample.copy()
    profile_df["trip_miles"]          = np.expm1(profile_df["log_trip_miles"])
    profile_df["base_passenger_fare"] = np.expm1(profile_df["log_base_passenger_fare"])
    profile_df["trip_duration_min"]   = np.expm1(profile_df["log_trip_duration_min"])

    PROFILE_COLS = ["trip_miles","base_passenger_fare","trip_duration_min",
                    "is_airport_trip","is_congestion_zone","has_tip","hour_of_day"]
    profile = profile_df.groupby("cluster")[PROFILE_COLS].agg(["mean","median"])
    profile.columns = ["_".join(c) for c in profile.columns]

    def assign_label(row):
        miles   = row["trip_miles_mean"]
        fare    = row["base_passenger_fare_mean"]
        airport = row["is_airport_trip_mean"]
        cong    = row["is_congestion_zone_mean"]
        if airport > 0.3:              return "Airport Fee Trips (Long-Haul)"
        if miles > 6:                  return "Cross-Borough Long"
        if cong >= 0.99 and miles < 3: return "Congestion Zone Short"
        if cong >= 0.99 and miles >= 3:return "Congestion Zone Mid"
        if cong < 0.05 and miles > 3:  return "Non-Zone Mid-Distance"
        if cong < 0.05 and miles <= 3: return "Non-Zone Short"
        if fare < 10:                  return "Minimum Fare Micro-Trip"
        return "Unclassified"

    profile["segment_label"] = [assign_label(profile.iloc[i]) for i in range(K)]

    print("\n  Cluster Profiles:")
    print(f"  {'Cluster':<10} {'Segment':<35} {'Miles':>6} {'Fare':>8} "
          f"{'Dur':>6} {'Airport':>8} {'Congst':>7} {'Tip':>5}")
    print(f"  {'-'*95}")
    for i in range(K):
        r = profile.iloc[i]
        print(f"  C{i:<9} {r['segment_label']:<35} "
              f"{r['trip_miles_mean']:>6.1f} "
              f"${r['base_passenger_fare_mean']:>7.2f} "
              f"{r['trip_duration_min_mean']:>6.1f} "
              f"{r['is_airport_trip_mean']:>8.2f} "
              f"{r['is_congestion_zone_mean']:>7.2f} "
              f"{r['has_tip_mean']:>5.2f}")

    # Cluster sizes
    sizes = pd.Series(labels).value_counts().sort_index()

    # Plot 1 — Profiles
    mean_cols = [c for c in profile.columns if c.endswith("_mean")]
    plot_metrics = [
        ("trip_miles_mean","Avg Trip Miles"),
        ("base_passenger_fare_mean","Avg Fare ($)"),
        ("trip_duration_min_mean","Avg Duration (min)"),
        ("is_airport_trip_mean","Airport Trip Rate"),
        ("is_congestion_zone_mean","Congestion Zone Rate"),
        ("has_tip_mean","Tip Rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16,9))
    axes = axes.flatten()
    xlabels = [f"C{i}\n{profile['segment_label'].iloc[i]}" for i in range(K)]
    for i, (col, title) in enumerate(plot_metrics):
        vals = profile[col].values
        bars = axes[i].bar(range(K), vals, color=COLORS[:K], alpha=0.85, edgecolor="white")
        axes[i].set_title(title, fontsize=10, fontweight="bold")
        axes[i].set_xticks(range(K))
        axes[i].set_xticklabels(xlabels, fontsize=7)
        axes[i].grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+max(vals)*0.01,
                         f"{val:.2f}", ha="center", fontsize=7)
    plt.suptitle("Cluster Profiles — Key Metrics by Segment",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("evaluation_output/ev_01_cluster_profiles.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2 — Radar
    radar_cols = ["trip_miles_mean","base_passenger_fare_mean",
                  "is_airport_trip_mean","is_congestion_zone_mean",
                  "has_tip_mean","trip_duration_min_mean"]
    radar_labels = ["Miles","Fare","Airport","Congestion","Tip","Duration"]
    radar_data = profile[radar_cols].copy()
    for col in radar_cols:
        mn, mx = radar_data[col].min(), radar_data[col].max()
        if mx > mn:
            radar_data[col] = (radar_data[col]-mn)/(mx-mn)
    N = len(radar_labels)
    angles = [n/float(N)*2*3.14159 for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(9,9), subplot_kw=dict(polar=True))
    for i in range(K):
        vals = radar_data.iloc[i].values.tolist() + [radar_data.iloc[i].values[0]]
        ax.plot(angles, vals, lw=2, color=COLORS[i],
                label=f"C{i}: {profile['segment_label'].iloc[i]}")
        ax.fill(angles, vals, alpha=0.08, color=COLORS[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11)
    ax.set_title("Cluster Radar Chart (normalised features)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    plt.tight_layout()
    plt.savefig("evaluation_output/ev_02_radar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3 — Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(16,6))
    for ax, col, title in zip(axes,
        ["trip_miles","base_passenger_fare","trip_duration_min"],
        ["Trip Miles","Base Fare ($)","Duration (min)"]):
        data_by_cl = [profile_df[profile_df["cluster"]==cl][col].values
                      for cl in range(K)]
        bp = ax.boxplot(data_by_cl, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker=".", markersize=1, alpha=0.2))
        for patch, color in zip(bp["boxes"], COLORS[:K]):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticklabels(
            [f"C{i}\n{profile['segment_label'].iloc[i]}" for i in range(K)],
            fontsize=6)
        ax.grid(True, alpha=0.2, axis="y")
    plt.suptitle("Within-Cluster Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("evaluation_output/ev_03_feature_boxplots.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4 — Pie
    fig, ax = plt.subplots(figsize=(8,8))
    seg_labels = [f"C{i}: {profile['segment_label'].iloc[i]}\n({sizes[i]:,})"
                  for i in range(K)]
    ax.pie(sizes.values, labels=seg_labels, colors=COLORS[:K],
           autopct="%1.1f%%", startangle=140,
           textprops={"fontsize":9},
           wedgeprops={"edgecolor":"white","linewidth":1.5})
    ax.set_title("Cluster Size Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("evaluation_output/ev_04_size_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()



