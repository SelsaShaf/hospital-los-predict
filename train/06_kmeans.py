import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH  = "models"
IMG_PATH    = "static/img"
os.makedirs(IMG_PATH, exist_ok=True)
K_OPTIMAL   = 3
K_RANGE     = range(2, 11)

plt.rcParams["figure.dpi"]        = 150
plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ============================================================
# MULAI
# ============================================================
print("=" * 55)
print("  06 K-MEANS — Prediksi Lama Tinggal Pasien RS")
print("=" * 55)
print()

# ============================================================
# LOAD DATA
# ============================================================
print("[1/5] Memuat data hasil preprocessing...")
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
    f"{MODEL_PATH}/data_split.pkl"
)
X_all = np.vstack([X_train.values, X_val.values, X_test.values])
print(f"      Total data       : {len(X_all):,} baris")
print()

# ============================================================
# ELBOW METHOD
# ============================================================
print("[2/5] Menghitung Elbow Method...")
inertias    = []
silhouettes = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_all, km.labels_, sample_size=5000, random_state=42)
    silhouettes.append(sil)
    print(f"      k={k}  Inertia={km.inertia_:.0f}  Silhouette={sil:.4f}")
print()

# ============================================================
# TRAINING K OPTIMAL
# ============================================================
print(f"[3/5] Training K-Means dengan k={K_OPTIMAL}...")
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans.fit(X_all)
labels      = kmeans.labels_
inertia     = kmeans.inertia_
sil_score   = silhouette_score(X_all, labels, sample_size=5000, random_state=42)

print(f"      Inertia          : {inertia:.2f}")
print(f"      Silhouette Score : {sil_score:.4f}")
print()

for i in range(K_OPTIMAL):
    count = np.sum(labels == i)
    print(f"      Cluster {i}        : {count:,} pasien ({count/len(labels)*100:.1f}%)")
print()

metrics = {
    "inertia": inertia,
    "silhouette": sil_score,
    "k": K_OPTIMAL
}
joblib.dump(metrics,  f"{MODEL_PATH}/metrics_kmeans.pkl")
joblib.dump(kmeans,   f"{MODEL_PATH}/kmeans_model.pkl")
joblib.dump({"inertias": inertias, "silhouettes": silhouettes, "k_range": list(K_RANGE)},
            f"{MODEL_PATH}/kmeans_elbow_data.pkl")
print("      Metrik tersimpan : models/metrics_kmeans.pkl")
print("      Model tersimpan  : models/kmeans_model.pkl")
print()

# ============================================================
# GRAFIK
# ============================================================
print("[4/5] Membuat grafik...")

# Elbow Method
print("      Grafik 1/2: Elbow Method...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(K_RANGE), inertias, marker="o", color="#2563EB",
        linewidth=1.5, markersize=6)
ax.axvline(x=K_OPTIMAL, color="#EF4444", linestyle="--",
           linewidth=1.5, label=f"k optimal = {K_OPTIMAL}")
ax.set_title("K-Means — Elbow Method", fontsize=12, fontweight="bold")
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("Inertia")
ax.set_xticks(list(K_RANGE))
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/kmeans_elbow.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: kmeans_elbow.png")

# Visualisasi Cluster PCA
print("      Grafik 2/2: Visualisasi cluster (PCA 2D)...")
sample_idx = np.random.choice(len(X_all), size=3000, replace=False)
X_sample   = X_all[sample_idx]
labels_sample = labels[sample_idx]

pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_sample)
colors  = ["#2563EB", "#10B981", "#F59E0B"]
labels_name = ["Rawat Singkat", "Rawat Sedang", "Rawat Lama"]

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(K_OPTIMAL):
    mask = labels_sample == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], label=labels_name[i],
               alpha=0.5, s=10)
ax.set_title("K-Means — Visualisasi Cluster (PCA 2D)", fontsize=12, fontweight="bold")
ax.set_xlabel("PCA Komponen 1")
ax.set_ylabel("PCA Komponen 2")
ax.legend()
ax.text(0.02, 0.95, f"Silhouette Score = {sil_score:.4f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/kmeans_cluster.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: kmeans_cluster.png")
print()

# ============================================================
# SELESAI
# ============================================================
print("=" * 55)
print(f"  K-Means selesai.")
print(f"  k={K_OPTIMAL} | Inertia={inertia:.2f} | Silhouette={sil_score:.4f}")
print("=" * 55)