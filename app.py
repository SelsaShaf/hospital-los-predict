import os
import io
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_file

# ============================================================
# INISIALISASI
# ============================================================
app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# BACKPROP NETWORK CLASS (wajib ada agar pkl bisa diload)
# ============================================================
class BackpropNetwork:
    def __init__(self, n_input, n_h1, n_h2, lr=0.01):
        self.lr = lr
        np.random.seed(42)
        self.W1 = np.random.randn(n_input, n_h1) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, n_h1))
        self.W2 = np.random.randn(n_h1,   n_h2) * np.sqrt(2.0 / n_h1)
        self.b2 = np.zeros((1, n_h2))
        self.W3 = np.random.randn(n_h2,   1)    * np.sqrt(2.0 / n_h2)
        self.b3 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = X  @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = np.maximum(0, self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        return self.Z3

    def backward(self, X, y_true, y_pred):
        pass

    def predict(self, X):
        return self.forward(X).flatten()

# ============================================================
# LOAD SEMUA MODEL & ARTEFAK
# ============================================================
def load_models():
    models  = {}
    metrics = {}
    try:
        models["linear"]   = joblib.load(os.path.join(MODEL_DIR, "linear_regression.pkl"))
        models["scaler"]   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        models["enc"]      = joblib.load(os.path.join(MODEL_DIR, "encoding_maps.pkl"))
        models["feats"]    = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        models["kmeans"]   = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
        models["backprop"] = joblib.load(os.path.join(MODEL_DIR, "backprop_model.pkl"))

        from tensorflow.keras.models import load_model
        models["ann"]    = load_model(os.path.join(MODEL_DIR, "ann_model.keras"))
        models["lstm"]   = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
        models["window"] = joblib.load(os.path.join(MODEL_DIR, "lstm_window_size.pkl"))

        metrics["linear"]   = joblib.load(os.path.join(MODEL_DIR, "metrics_linear_regression.pkl"))
        metrics["ann"]      = joblib.load(os.path.join(MODEL_DIR, "metrics_ann.pkl"))
        metrics["lstm"]     = joblib.load(os.path.join(MODEL_DIR, "metrics_lstm.pkl"))
        metrics["kmeans"]   = joblib.load(os.path.join(MODEL_DIR, "metrics_kmeans.pkl"))
        metrics["backprop"] = joblib.load(os.path.join(MODEL_DIR, "metrics_backprop.pkl"))

        ann_hist  = joblib.load(os.path.join(MODEL_DIR, "ann_history.pkl"))
        lstm_hist = joblib.load(os.path.join(MODEL_DIR, "lstm_history.pkl"))
        elbow     = joblib.load(os.path.join(MODEL_DIR, "kmeans_elbow_data.pkl"))

        print("[OK] Semua model berhasil dimuat.")
        return models, metrics, ann_hist, lstm_hist, elbow

    except Exception as e:
        print(f"[WARN] Model belum tersedia: {e}")
        return models, metrics, {}, {}, {}

MODELS, METRICS, ANN_HIST, LSTM_HIST, ELBOW_DATA = load_models()

# ============================================================
# HELPER — PREPROCESSING INPUT
# ============================================================
def preprocess_input(data: dict) -> np.ndarray:
    enc  = MODELS.get("enc", {})
    feat = MODELS.get("feats", [])

    rcount_map = enc.get("rcount", {"0":0,"1":1,"2":2,"3":3,"4":4,"5+":5})
    gender_map = enc.get("gender", {"F":0,"M":1})
    facid_map  = enc.get("facid",  {"A":0,"B":1,"C":2,"D":3,"E":4})

    row = {
        "rcount":                    rcount_map.get(str(data.get("rcount", "0")), 0),
        "gender":                    gender_map.get(str(data.get("gender", "F")), 0),
        "dialysisrenalendstage":     int(data.get("dialysisrenalendstage", 0)),
        "asthma":                    int(data.get("asthma", 0)),
        "irondef":                   int(data.get("irondef", 0)),
        "pneum":                     int(data.get("pneum", 0)),
        "substancedependence":       int(data.get("substancedependence", 0)),
        "psychologicaldisordermajor":int(data.get("psychologicaldisordermajor", 0)),
        "depress":                   int(data.get("depress", 0)),
        "psychother":                int(data.get("psychother", 0)),
        "fibrosisandother":          int(data.get("fibrosisandother", 0)),
        "malnutrition":              int(data.get("malnutrition", 0)),
        "hemo":                      int(data.get("hemo", 0)),
        "hematocrit":                float(data.get("hematocrit", 40.0)),
        "neutrophils":               float(data.get("neutrophils", 60.0)),
        "sodium":                    float(data.get("sodium", 138.0)),
        "glucose":                   float(data.get("glucose", 100.0)),
        "bloodureanitro":            float(data.get("bloodureanitro", 14.0)),
        "creatinine":                float(data.get("creatinine", 1.0)),
        "bmi":                       float(data.get("bmi", 25.0)),
        "pulse":                     int(data.get("pulse", 80)),
        "respiration":               float(data.get("respiration", 16.0)),
        "secondarydiagnosisnonicd9": int(data.get("secondarydiagnosisnonicd9", 0)),
        "facid":                     facid_map.get(str(data.get("facid", "A")), 0),
    }

    df_row = pd.DataFrame([row])
    if feat:
        df_row = df_row[feat]

    scaler   = MODELS.get("scaler")
    X_scaled = scaler.transform(df_row) if scaler else df_row.values
    return X_scaled


def get_category(days: float) -> dict:
    days = round(days)
    if days <= 3:
        return {"label": "Rawat Singkat", "color": "success",
                "desc": "Pasien kemungkinan dapat segera dipulangkan dalam waktu dekat."}
    elif days <= 7:
        return {"label": "Rawat Sedang", "color": "warning",
                "desc": "Pasien membutuhkan pemantauan beberapa hari sebelum dipulangkan."}
    else:
        return {"label": "Rawat Lama", "color": "danger",
                "desc": "Pasien membutuhkan perawatan intensif lebih lanjut."}

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def home():
    best_model = "—"
    best_rmse  = "—"
    best_r2    = "—"
    if METRICS:
        supervised = {k: v for k, v in METRICS.items() if k != "kmeans"}
        if supervised:
            bk = min(supervised, key=lambda k: supervised[k]["RMSE"])
            name_map = {"linear":"Linear Regression","ann":"ANN",
                        "lstm":"LSTM","backprop":"Backpropagation"}
            best_model = name_map.get(bk, bk)
            best_rmse  = f"{supervised[bk]['RMSE']:.4f}"
            best_r2    = f"{supervised[bk]['R2']:.4f}"

    return render_template("home.html",
                           best_model=best_model,
                           best_rmse=best_rmse,
                           best_r2=best_r2)


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data     = request.get_json()
        X_scaled = preprocess_input(data)

        results = {}

        if "linear" in MODELS:
            pred = float(MODELS["linear"].predict(X_scaled)[0])
            results["linear"] = max(1, round(pred))

        if "ann" in MODELS:
            pred = float(MODELS["ann"].predict(X_scaled, verbose=0)[0][0])
            results["ann"] = max(1, round(pred))

        if "lstm" in MODELS:
            window = MODELS.get("window", 1)
            X_seq  = X_scaled.reshape(1, window, X_scaled.shape[1])
            pred   = float(MODELS["lstm"].predict(X_seq, verbose=0)[0][0])
            results["lstm"] = max(1, round(pred))

        if "backprop" in MODELS:
            pred = float(MODELS["backprop"].predict(X_scaled)[0])
            results["backprop"] = max(1, round(pred))

        if "kmeans" in MODELS:
            cluster = int(MODELS["kmeans"].predict(X_scaled)[0])
            cluster_map = {0: "Rawat Singkat", 1: "Rawat Sedang", 2: "Rawat Lama"}
            results["kmeans_cluster"] = cluster_map.get(cluster, f"Cluster {cluster}")

        best_pred = results.get("lstm") or results.get("ann") or results.get("linear", 1)
        category  = get_category(best_pred)
        confidence_low  = max(1, best_pred - 1)
        confidence_high = best_pred + 1

        return jsonify({
            "status":          "ok",
            "prediction":      best_pred,
            "confidence_low":  confidence_low,
            "confidence_high": confidence_high,
            "category":        category,
            "all_models":      results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict-csv", methods=["POST"])
def api_predict_csv():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "error", "message": "File tidak ditemukan"}), 400

        df = pd.read_csv(file)

        enc    = MODELS.get("enc", {})
        feat   = MODELS.get("feats", [])
        scaler = MODELS.get("scaler")

        rcount_map = enc.get("rcount", {"0":0,"1":1,"2":2,"3":3,"4":4,"5+":5})
        gender_map = enc.get("gender", {"F":0,"M":1})
        facid_map  = enc.get("facid",  {"A":0,"B":1,"C":2,"D":3,"E":4})

        df["rcount"] = df["rcount"].astype(str).map(rcount_map).fillna(0)
        df["gender"] = df["gender"].astype(str).map(gender_map).fillna(0)
        df["facid"]  = df["facid"].astype(str).map(facid_map).fillna(0)

        drop_cols = [c for c in ["eid","vdate","discharged","lengthofstay"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        if feat:
            for mc in [f for f in feat if f not in df.columns]:
                df[mc] = 0
            df = df[feat]

        X = scaler.transform(df) if scaler else df.values

        preds = []
        if "lstm" in MODELS:
            window = MODELS.get("window", 1)
            for row in X:
                X_seq = row.reshape(1, window, len(row))
                p = float(MODELS["lstm"].predict(X_seq, verbose=0)[0][0])
                preds.append(max(1, round(p)))
        elif "linear" in MODELS:
            preds = [max(1, round(float(p))) for p in MODELS["linear"].predict(X)]

        df["prediksi_hari"] = preds
        df["kategori"]      = df["prediksi_hari"].apply(lambda d: get_category(d)["label"])

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="hasil_prediksi.csv"
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sample")
def api_sample():
    try:
        df  = pd.read_csv(os.path.join(BASE_DIR, "data", "LengthOfStay.csv"))
        row = df.sample(1).iloc[0]
        return jsonify({
            "status": "ok",
            "data": {
                "rcount":                     str(row["rcount"]),
                "gender":                     str(row["gender"]),
                "facid":                      str(row["facid"]),
                "hematocrit":                 round(float(row["hematocrit"]), 2),
                "neutrophils":                round(float(row["neutrophils"]), 2),
                "sodium":                     round(float(row["sodium"]), 2),
                "glucose":                    round(float(row["glucose"]), 2),
                "bloodureanitro":             round(float(row["bloodureanitro"]), 2),
                "creatinine":                 round(float(row["creatinine"]), 2),
                "bmi":                        round(float(row["bmi"]), 2),
                "pulse":                      int(row["pulse"]),
                "respiration":                round(float(row["respiration"]), 2),
                "dialysisrenalendstage":      int(row["dialysisrenalendstage"]),
                "asthma":                     int(row["asthma"]),
                "irondef":                    int(row["irondef"]),
                "pneum":                      int(row["pneum"]),
                "substancedependence":        int(row["substancedependence"]),
                "psychologicaldisordermajor": int(row["psychologicaldisordermajor"]),
                "depress":                    int(row["depress"]),
                "psychother":                 int(row["psychother"]),
                "fibrosisandother":           int(row["fibrosisandother"]),
                "malnutrition":               int(row["malnutrition"]),
                "hemo":                       int(row["hemo"]),
                "secondarydiagnosisnonicd9":  int(row["secondarydiagnosisnonicd9"]),
                "actual_los":                 int(row["lengthofstay"])
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/compare")
def compare():
    metrics_json = {}
    if METRICS:
        for k, v in METRICS.items():
            metrics_json[k] = {key: round(val, 4) for key, val in v.items()}

    ann_loss  = ANN_HIST.get("loss",     [])
    ann_val   = ANN_HIST.get("val_loss", [])
    lstm_loss = LSTM_HIST.get("loss",     [])
    lstm_val  = LSTM_HIST.get("val_loss", [])
    elbow_k   = ELBOW_DATA.get("k_range",  [])
    elbow_in  = ELBOW_DATA.get("inertias", [])

    return render_template("compare.html",
                           metrics=metrics_json,
                           ann_loss=ann_loss,
                           ann_val=ann_val,
                           lstm_loss=lstm_loss,
                           lstm_val=lstm_val,
                           elbow_k=elbow_k,
                           elbow_in=elbow_in)


@app.route("/dataset")
def dataset():
    stats   = {}
    preview = []
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "LengthOfStay.csv"))
        num_cols = ["hematocrit","neutrophils","sodium","glucose",
                    "bloodureanitro","creatinine","bmi","pulse",
                    "respiration","lengthofstay"]
        for col in num_cols:
            stats[col] = {
                "min":  round(float(df[col].min()),  2),
                "mean": round(float(df[col].mean()), 2),
                "max":  round(float(df[col].max()),  2),
                "std":  round(float(df[col].std()),  2),
            }
        preview = df.head(10).to_dict(orient="records")
    except Exception as e:
        print(f"[WARN] Dataset error: {e}")

    return render_template("dataset.html", stats=stats, preview=preview)


@app.route("/about")
def about():
    return render_template("about.html")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[OK] Server berjalan di http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)