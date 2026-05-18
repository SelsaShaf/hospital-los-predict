import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "models"
IMG_PATH   = "static/img"
os.makedirs(IMG_PATH, exist_ok=True)

plt.rcParams["figure.dpi"]        = 150
plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ============================================================
# MULAI
# ============================================================
print("=" * 55)
print("  03 LINEAR REGRESSION — Prediksi Lama Tinggal Pasien RS")
print("=" * 55)
print()

# ============================================================
# LOAD DATA
# ============================================================
print("[1/4] Memuat data hasil preprocessing...")
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
    f"{MODEL_PATH}/data_split.pkl"
)
print(f"      Train  : {len(X_train):,} baris")
print(f"      Val    : {len(X_val):,} baris")
print(f"      Test   : {len(X_test):,} baris")
print()

# ============================================================
# TRAINING
# ============================================================
print("[2/4] Melatih model Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)
print("      Training selesai.")
print()

# ============================================================
# EVALUASI
# ============================================================
print("[3/4] Evaluasi model...")

y_pred_train = model.predict(X_train)
y_pred_val   = model.predict(X_val)
y_pred_test  = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

mae_val  = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val   = r2_score(y_val, y_pred_val)

print(f"      --- Validasi ---")
print(f"      MAE              : {mae_val:.4f}")
print(f"      RMSE             : {rmse_val:.4f}")
print(f"      R²               : {r2_val:.4f}")
print()
print(f"      --- Test ---")
print(f"      MAE              : {mae:.4f}")
print(f"      RMSE             : {rmse:.4f}")
print(f"      R²               : {r2:.4f}")
print()

# Simpan metrik
metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
joblib.dump(metrics, f"{MODEL_PATH}/metrics_linear_regression.pkl")
print("      Metrik tersimpan : models/metrics_linear_regression.pkl")
print()

# ============================================================
# SIMPAN MODEL
# ============================================================
joblib.dump(model, f"{MODEL_PATH}/linear_regression.pkl")
print("      Model tersimpan  : models/linear_regression.pkl")
print()

# ============================================================
# GRAFIK
# ============================================================
print("[4/4] Membuat grafik...")

# Residual plot
print("      Grafik 1/1: Residual plot (actual vs predicted)...")
sample_idx = np.random.choice(len(y_test), size=1000, replace=False)
y_test_arr  = np.array(y_test)[sample_idx]
y_pred_arr  = y_pred_test[sample_idx]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test_arr, y_pred_arr, alpha=0.4, color="#2563EB", s=15)
mn = min(y_test_arr.min(), y_pred_arr.min())
mx = max(y_test_arr.max(), y_pred_arr.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Ideal")
ax.set_title("Linear Regression — Actual vs Predicted", fontsize=12, fontweight="bold")
ax.set_xlabel("Actual (Hari)")
ax.set_ylabel("Predicted (Hari)")
ax.legend()
ax.text(0.05, 0.90, f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/linear_residual_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: linear_residual_plot.png")
print()

# ============================================================
# SELESAI
# ============================================================
print("=" * 55)
print(f"  Linear Regression selesai.")
print(f"  MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
print("=" * 55)