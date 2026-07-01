import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

IN_SCALED  = "../output/scaling_output/scaled.parquet"
IN_SCALER  = "../output/scaling_output/scaler.pkl"
OUT_DIR    = "../output/modelling_output"
K_RANGE    = range(2, 11)
SEED       = 42
os.makedirs(OUT_DIR, exist_ok=True)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)


# LOAD
df = pd.read_parquet(IN_SCALED)
X  = df.values
log(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} features")
log(f"  Features: {list(df.columns)}")


# 1. ELBOW + SILHOUETTE ANALYSIS
log(f"\n-- 1. Elbow + Silhouette Analysis (k = {min(K_RANGE)} to {max(K_RANGE)}) --")
log(f"  n_init=10 (best of 10 random initialisations per k)")
log(f"  silhouette sample_size=20,000 (for speed)\n")
log(f"  {'k':>4}  {'inertia':>12}  {'silhouette':>12}  {'delta_inertia':>14}")
log(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*14}")

results = []
prev_inertia = None

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    sil = silhouette_score(X, labels, sample_size=20_000, random_state=SEED)
    delta = (prev_inertia - inertia) if prev_inertia else None
    delta_str = f"{delta:>14,.0f}" if delta else "             —"
    log(f"  {k:>4}  {inertia:>12,.0f}  {sil:>12.4f}  {delta_str}")
    results.append({"k": k, "inertia": inertia, "silhouette": sil})
    prev_inertia = inertia

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/elbow_results.csv", index=False)
log(f"\n  -> Saved elbow_results.csv")


# 2. ELBOW PLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("K Selection: Elbow Curve + Silhouette Score",
             fontsize=13, fontweight="bold")

# Inertia
ax1.plot(results_df["k"], results_df["inertia"],
         "bo-", linewidth=2, markersize=8)
ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
ax1.set_ylabel("Inertia (WCSS)", fontsize=11)
ax1.set_title("Elbow Curve", fontsize=11, fontweight="bold")
ax1.set_xticks(list(K_RANGE))
ax1.grid(True, alpha=0.3)

# Silhouette
ax2.plot(results_df["k"], results_df["silhouette"],
         "rs-", linewidth=2, markersize=8)
ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
ax2.set_ylabel("Silhouette Score", fontsize=11)
ax2.set_title("Silhouette Score (higher = better)", fontsize=11,
              fontweight="bold")
ax2.set_xticks(list(K_RANGE))
ax2.grid(True, alpha=0.3)

# Mark best silhouette
best_sil_k = results_df.loc[results_df["silhouette"].idxmax(), "k"]
best_sil   = results_df["silhouette"].max()
ax2.axvline(best_sil_k, color="green", linestyle="--", alpha=0.7,
            label=f"best k={best_sil_k} (sil={best_sil:.3f})")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mod_01_elbow.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"  -> Saved mod_01_elbow.png")
log(f"\n  Best silhouette at k={best_sil_k} (score={best_sil:.4f})")


# 3. CHOOSE K AND FIT FINAL MODEL


CHOSEN_K = 7
log(f"  Chosen k={CHOSEN_K}")
log(f"  Silhouette at k=3: {results_df.loc[results_df['k']==3, 'silhouette'].values[0]:.4f}")

final_model = KMeans(n_clusters=CHOSEN_K, random_state=SEED, n_init=20)
final_labels = final_model.fit_predict(X)

final_sil = silhouette_score(X, final_labels,sample_size=20_000, random_state=SEED)
log(f"\n  Final model: k={CHOSEN_K}")
log(f"  Inertia : {final_model.inertia_:,.0f}")
log(f"  Silhouette : {final_sil:.4f}")
log(f"  Cluster sizes  :")
unique, counts = np.unique(final_labels, return_counts=True)
for cl, cnt in zip(unique, counts):
    log(f"    Cluster {cl}: {cnt:>7,}  ({cnt/len(final_labels)*100:.1f}%)")


# 4. PCA VISUALISATION (2D projection of clusters)
log(f"\n-- 3. PCA 2D Visualisation --")
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X)
var_explained = pca.explained_variance_ratio_
log(f"  PC1 variance explained: {var_explained[0]*100:.1f}%")
log(f"  PC2 variance explained: {var_explained[1]*100:.1f}%")
log(f"  Total : {sum(var_explained)*100:.1f}%")

COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0",
          "#FF9800","#00BCD4","#E91E63","#795548","#607D8B","#CDDC39"]

# Subsample for plotting speed
plot_idx = np.random.default_rng(SEED).choice(len(X_pca), 50_000,
                                               replace=False)
fig, ax = plt.subplots(figsize=(10, 7))
for cl in range(CHOSEN_K):
    mask = final_labels[plot_idx] == cl
    ax.scatter(X_pca[plot_idx][mask, 0],
               X_pca[plot_idx][mask, 1],
               s=2, alpha=0.35,
               color=COLORS[cl % len(COLORS)],
               label=f"Cluster {cl} (n={counts[cl]:,})")

ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=11)
ax.set_title(f"K-Means Clusters (k={CHOSEN_K}) — PCA 2D Projection",
             fontsize=12, fontweight="bold")
ax.legend(markerscale=6, fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mod_02_clusters_2d.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"  -> Saved mod_02_clusters_2d.png")


# 5. SAVE MODEL AND LABELS
with open(f"{OUT_DIR}/model.pkl", "wb") as f:
    pickle.dump({"model": final_model, "k": CHOSEN_K,
                 "features": list(df.columns)}, f)
log(f"\n  -> Saved model.pkl")

pd.DataFrame({"cluster": final_labels}).to_parquet(
    f"{OUT_DIR}/labels.parquet", index=False)
log(f"  -> Saved labels.parquet")

with open(f"{OUT_DIR}/modelling_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"  -> Saved modelling_report.txt")
