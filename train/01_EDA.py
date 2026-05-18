import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# KONFIGURASI
# ============================================================
DATA_PATH = "data/LengthOfStay.csv"
IMG_PATH  = "static/img"
os.makedirs(IMG_PATH, exist_ok=True)

plt.rcParams["figure.dpi"]    = 150
plt.rcParams["font.family"]   = "sans-serif"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ============================================================
# MULAI
# ============================================================
print("=" * 55)
print("  01 EDA — Prediksi Lama Tinggal Pasien RS")
print("=" * 55)
print()

# ============================================================
# LOAD DATA
# ============================================================
print("[1/6] Memuat dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Berhasil dimuat.")
print(f"      Jumlah baris     : {df.shape[0]:,}")
print(f"      Jumlah kolom     : {df.shape[1]}")
print()

# ============================================================
# INFO KOLOM
# ============================================================
print("[2/6] Informasi kolom:")
for col in df.columns:
    print(f"      {col:35} {str(df[col].dtype)}")
print()

# ============================================================
# MISSING VALUES
# ============================================================
print("[3/6] Mengecek missing values...")
missing = df.isnull().sum()
total_missing = missing.sum()
if total_missing == 0:
    print("      Missing values   : Tidak ada")
else:
    print("      Kolom dengan missing values:")
    for col, val in missing[missing > 0].items():
        pct = val / len(df) * 100
        print(f"        - {col}: {val:,} ({pct:.2f}%)")
print()

# ============================================================
# CEK DUPLIKAT
# ============================================================
print("[4/6] Mengecek duplikat...")
dup = df.duplicated().sum()
print(f"      Duplikat         : {dup:,}")
print()

# ============================================================
# STATISTIK DESKRIPTIF
# ============================================================
print("[5/6] Statistik deskriptif target (lengthofstay):")
los = df["lengthofstay"]
print(f"      Min              : {los.min()} hari")
print(f"      Maksimum         : {los.max()} hari")
print(f"      Rata-rata        : {los.mean():.2f} hari")
print(f"      Median           : {los.median():.0f} hari")
print(f"      Std Deviasi      : {los.std():.2f}")
print()
print("      Nilai unik rcount :", sorted(df["rcount"].unique()))
print("      Nilai unik gender :", sorted(df["gender"].unique()))
print("      Nilai unik facid  :", sorted(df["facid"].unique()))
print()

# ============================================================
# GENERATE GRAFIK
# ============================================================
print("[6/6] Membuat grafik EDA...")

# Grafik 1 — Distribusi lengthofstay
print("      Grafik 1/4: Distribusi lama rawat...")
fig, ax = plt.subplots(figsize=(8, 5))
counts = df["lengthofstay"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color="#2563EB", alpha=0.85, width=0.7)
ax.set_title("Distribusi Lama Rawat Inap Pasien", fontsize=13, fontweight="bold")
ax.set_xlabel("Lama Rawat (Hari)")
ax.set_ylabel("Jumlah Pasien")
ax.set_xticks(range(1, 18))
for i, v in zip(counts.index, counts.values):
    ax.text(i, v + 200, str(v), ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/eda_distribusi_los.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: eda_distribusi_los.png")

# Grafik 2 — Heatmap korelasi
print("      Grafik 2/4: Heatmap korelasi...")
num_cols = ["hematocrit", "neutrophils", "sodium", "glucose",
            "bloodureanitro", "creatinine", "bmi", "pulse",
            "respiration", "lengthofstay"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, mask=mask,
            annot_kws={"size": 8})
ax.set_title("Heatmap Korelasi Fitur Numerik", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/eda_korelasi_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: eda_korelasi_heatmap.png")

# Grafik 3 — Distribusi gender
print("      Grafik 3/4: Distribusi gender...")
fig, ax = plt.subplots(figsize=(6, 4))
gender_counts = df["gender"].value_counts()
bars = ax.bar(["Perempuan (F)", "Laki-laki (M)"],
              [gender_counts.get("F", 0), gender_counts.get("M", 0)],
              color=["#93C5FD", "#2563EB"], alpha=0.85, width=0.5)
ax.set_title("Distribusi Gender Pasien", fontsize=13, fontweight="bold")
ax.set_ylabel("Jumlah Pasien")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 200,
            f"{bar.get_height():,}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/eda_distribusi_gender.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: eda_distribusi_gender.png")

# Grafik 4 — Boxplot lengthofstay
print("      Grafik 4/4: Boxplot lama rawat...")
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot(df["lengthofstay"], patch_artist=True,
                medianprops=dict(color="white", linewidth=2))
bp["boxes"][0].set_facecolor("#2563EB")
bp["boxes"][0].set_alpha(0.8)
ax.set_title("Boxplot Lama Rawat Inap", fontsize=13, fontweight="bold")
ax.set_ylabel("Lama Rawat (Hari)")
ax.set_xticks([])
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/eda_boxplot_los.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: eda_boxplot_los.png")

# ============================================================
# SELESAI
# ============================================================
print()
print("=" * 55)
print("  EDA selesai. Grafik tersimpan di static/img/")
print("=" * 55)