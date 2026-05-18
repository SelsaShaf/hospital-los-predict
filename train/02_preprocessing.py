import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ============================================================
# KONFIGURASI
# ============================================================
DATA_PATH  = "data/LengthOfStay.csv"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

# ============================================================
# MULAI
# ============================================================
print("=" * 55)
print("  02 PREPROCESSING — Prediksi Lama Tinggal Pasien RS")
print("=" * 55)
print()

# ============================================================
# LOAD DATA
# ============================================================
print("[1/7] Memuat dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Jumlah baris     : {df.shape[0]:,}")
print(f"      Jumlah kolom     : {df.shape[1]}")
print()

# ============================================================
# DROP KOLOM TIDAK RELEVAN
# ============================================================
print("[2/7] Menghapus kolom tidak relevan...")
drop_cols = ["eid", "vdate", "discharged"]
df = df.drop(columns=drop_cols)
print(f"      Kolom dihapus    : {drop_cols}")
print(f"      Kolom tersisa    : {df.shape[1]}")
print()

# ============================================================
# CEK MISSING VALUES
# ============================================================
print("[3/7] Mengecek missing values...")
missing = df.isnull().sum().sum()
if missing == 0:
    print("      Missing values   : Tidak ada")
else:
    print(f"      Missing values   : {missing:,} — melakukan dropna")
    df = df.dropna()
    print(f"      Baris setelah dropna : {df.shape[0]:,}")
print()

# ============================================================
# ENCODING KOLOM KATEGORIKAL
# ============================================================
print("[4/7] Encoding kolom kategorikal...")

# rcount — ordinal encoding agar urutan terjaga
rcount_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}
df["rcount"] = df["rcount"].map(rcount_map)
print(f"      rcount  : ordinal encoding {rcount_map}")

# gender — binary encoding
gender_map = {"F": 0, "M": 1}
df["gender"] = df["gender"].map(gender_map)
print(f"      gender  : binary encoding {gender_map}")

# facid — ordinal encoding A=0 B=1 C=2 D=3 E=4
facid_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
df["facid"] = df["facid"].map(facid_map)
print(f"      facid   : ordinal encoding {facid_map}")

encoding_maps = {
    "rcount": rcount_map,
    "gender": gender_map,
    "facid":  facid_map
}
joblib.dump(encoding_maps, f"{MODEL_PATH}/encoding_maps.pkl")
print(f"      Encoding maps tersimpan : models/encoding_maps.pkl")
print()

# ============================================================
# PISAHKAN FITUR DAN TARGET
# ============================================================
print("[5/7] Memisahkan fitur dan target...")
X = df.drop(columns=["lengthofstay"])
y = df["lengthofstay"]
feature_names = X.columns.tolist()
print(f"      Jumlah fitur     : {len(feature_names)}")
print(f"      Target           : lengthofstay")
print()

# ============================================================
# NORMALISASI
# ============================================================
print("[6/7] Normalisasi fitur (StandardScaler)...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=feature_names
)
joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
print(f"      Kolom dinormalisasi : {len(feature_names)} kolom")
print(f"      Scaler tersimpan    : models/scaler.pkl")
print()

# ============================================================
# SPLIT DATA 70 / 15 / 15
# ============================================================
print("[7/7] Membagi data train / validasi / test (70/15/15)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

print(f"      Total data       : {len(df):,}")
print(f"      Train            : {len(X_train):,} ({len(X_train)/len(df)*100:.0f}%)")
print(f"      Validasi         : {len(X_val):,}  ({len(X_val)/len(df)*100:.0f}%)")
print(f"      Test             : {len(X_test):,} ({len(X_test)/len(df)*100:.0f}%)")
print()

joblib.dump(
    (X_train, X_val, X_test,
     y_train, y_val, y_test),
    f"{MODEL_PATH}/data_split.pkl"
)
joblib.dump(feature_names, f"{MODEL_PATH}/feature_names.pkl")

print("      File tersimpan:")
print("        models/data_split.pkl")
print("        models/feature_names.pkl")
print()

# ============================================================
# SELESAI
# ============================================================
print("=" * 55)
print("  Preprocessing selesai.")
print("=" * 55)