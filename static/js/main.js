// ============================================================
// DARK MODE
// ============================================================
const html       = document.documentElement;
const darkToggle = document.getElementById("darkToggle");
const savedTheme = localStorage.getItem("theme") || "light";

html.setAttribute("data-bs-theme", savedTheme);
if (darkToggle) {
  darkToggle.innerHTML = savedTheme === "dark"
    ? '<i class="bi bi-sun"></i>'
    : '<i class="bi bi-moon"></i>';

  darkToggle.addEventListener("click", () => {
    const current = html.getAttribute("data-bs-theme");
    const next    = current === "dark" ? "light" : "dark";
    html.setAttribute("data-bs-theme", next);
    localStorage.setItem("theme", next);
    darkToggle.innerHTML = next === "dark"
      ? '<i class="bi bi-sun"></i>'
      : '<i class="bi bi-moon"></i>';
  });
}

// ============================================================
// AOS INIT
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  if (typeof AOS !== "undefined") {
    AOS.init({ duration: 600, once: true, offset: 60 });
  }
});

// ============================================================
// AUTO SCROLL KE FORM SAAT DARI HOME
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  if (window.location.hash === "#tab-awam") {
    const formEl = document.getElementById("tab-awam");
    if (formEl) {
      setTimeout(() => {
        formEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 300);
    }
  }
});

// ============================================================
// LOADING OVERLAY
// ============================================================
function showLoading() {
  const el = document.getElementById("loadingOverlay");
  if (el) el.classList.add("show");
}

function hideLoading() {
  const el = document.getElementById("loadingOverlay");
  if (el) el.classList.remove("show");
}

// ============================================================
// FORM VALIDATION — MODE TENAGA MEDIS
// ============================================================
function validateForm() {
  const numFields = [
    "hematocrit","neutrophils","sodium","glucose",
    "bloodureanitro","creatinine","bmi","pulse","respiration"
  ];
  let valid = true;
  numFields.forEach(id => {
    const el  = document.getElementById(id);
    const err = document.getElementById(id + "_err");
    if (!el) return;
    const val = parseFloat(el.value);
    el.classList.remove("is-invalid");
    if (isNaN(val) || el.value.trim() === "") {
      el.classList.add("is-invalid");
      if (err) err.textContent = "Field ini wajib diisi.";
      valid = false;
    }
  });
  return valid;
}

// ============================================================
// PREDICT — MODE TENAGA MEDIS
// ============================================================
const predictBtn = document.getElementById("predictBtn");
if (predictBtn) {
  predictBtn.addEventListener("click", async () => {
    if (!validateForm()) return;

    const data = {
      rcount:   document.getElementById("rcount")?.value  || "0",
      gender:   document.getElementById("gender")?.value  || "F",
      facid:    document.getElementById("facid")?.value   || "A",
      hematocrit:     parseFloat(document.getElementById("hematocrit")?.value),
      neutrophils:    parseFloat(document.getElementById("neutrophils")?.value),
      sodium:         parseFloat(document.getElementById("sodium")?.value),
      glucose:        parseFloat(document.getElementById("glucose")?.value),
      bloodureanitro: parseFloat(document.getElementById("bloodureanitro")?.value),
      creatinine:     parseFloat(document.getElementById("creatinine")?.value),
      bmi:            parseFloat(document.getElementById("bmi")?.value),
      pulse:          parseInt(document.getElementById("pulse")?.value),
      respiration:    parseFloat(document.getElementById("respiration")?.value),
      dialysisrenalendstage:      document.getElementById("dialysisrenalendstage")?.checked ? 1 : 0,
      asthma:                     document.getElementById("asthma")?.checked ? 1 : 0,
      irondef:                    document.getElementById("irondef")?.checked ? 1 : 0,
      pneum:                      document.getElementById("pneum")?.checked ? 1 : 0,
      substancedependence:        document.getElementById("substancedependence")?.checked ? 1 : 0,
      psychologicaldisordermajor: document.getElementById("psychologicaldisordermajor")?.checked ? 1 : 0,
      depress:                    document.getElementById("depress")?.checked ? 1 : 0,
      psychother:                 document.getElementById("psychother")?.checked ? 1 : 0,
      fibrosisandother:           document.getElementById("fibrosisandother")?.checked ? 1 : 0,
      malnutrition:               document.getElementById("malnutrition")?.checked ? 1 : 0,
      hemo:                       document.getElementById("hemo")?.checked ? 1 : 0,
      secondarydiagnosisnonicd9:  document.getElementById("secondarydiagnosisnonicd9")?.checked ? 1 : 0,
    };

    showLoading();
    try {
      const res  = await fetch("/api/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(data)
      });
      const json = await res.json();
      hideLoading();
      if (json.status === "ok") {
        json._mode = "medis";
        renderResult(json);
      } else {
        alert("Terjadi kesalahan: " + json.message);
      }
    } catch (e) {
      hideLoading();
      alert("Gagal menghubungi server.");
    }
  });
}

// ============================================================
// RENDER RESULT
// ============================================================
function renderResult(json) {
  const sec = document.getElementById("resultSection");
  if (!sec) return;
  sec.style.display = "block";
  sec.scrollIntoView({ behavior: "smooth", block: "start" });

  const resDays = document.getElementById("resDays");
  if (resDays) resDays.textContent = json.prediction;

  const resCatDesc = document.getElementById("resCatDesc");
  if (resCatDesc) resCatDesc.textContent = json.category.desc;

  const catBadge = document.getElementById("resCatBadge");
  if (catBadge) {
    catBadge.className   = "badge-custom badge-" + json.category.color;
    catBadge.textContent = json.category.label;
  }

  const resConfLow  = document.getElementById("resConfLow");
  const resConfHigh = document.getElementById("resConfHigh");
  if (resConfLow)  resConfLow.textContent  = json.confidence_low;
  if (resConfHigh) resConfHigh.textContent = json.confidence_high;

  const pct_low  = ((json.confidence_low  - 1) / 16 * 100).toFixed(1);
  const pct_high = ((json.confidence_high - 1) / 16 * 100).toFixed(1);
  const rangeEl  = document.getElementById("confRange");
  if (rangeEl) {
    rangeEl.style.left  = pct_low  + "%";
    rangeEl.style.width = (pct_high - pct_low) + "%";
  }

  const all   = json.all_models || {};
  const names = {
    linear:   "Linear Regression",
    ann:      "ANN",
    lstm:     "LSTM",
    backprop: "Backpropagation"
  };

  const maxVal = Math.max(...Object.entries(all)
    .filter(([k]) => k !== "kmeans_cluster")
    .map(([, v]) => v), 1);

  let rows = "";
  Object.entries(all).forEach(([key, val]) => {
    if (key === "kmeans_cluster") {
      rows += `<tr>
        <td>K-Means</td>
        <td colspan="2" class="text-muted">Cluster: ${val}</td>
      </tr>`;
      return;
    }
    const pct = (val / (maxVal + 1) * 100).toFixed(0);
    rows += `<tr>
      <td>${names[key] || key}</td>
      <td><div class="progress" style="height:8px">
        <div class="progress-bar" style="width:${pct}%;background:var(--primary)"></div>
      </div></td>
      <td style="font-weight:600;white-space:nowrap">${val} hari</td>
    </tr>`;
  });

  const tbody = document.getElementById("allModelsTbody");
  if (tbody) tbody.innerHTML = rows;

  const disclaimer = document.getElementById("awamDisclaimer");
  if (disclaimer) {
    disclaimer.style.display = json._mode === "awam" ? "block" : "none";
  }

  buildSuggestions(json);
}

function buildSuggestions(json) {
  const box = document.getElementById("suggestionBox");
  if (!box) return;
  const items = [];

  const glucoseEl = json._mode === "awam"
    ? parseFloat(document.getElementById("aw_glucose")?.value)
    : parseFloat(document.getElementById("glucose")?.value);
  const pulseEl = json._mode === "awam"
    ? parseFloat(document.getElementById("aw_pulse")?.value)
    : parseFloat(document.getElementById("pulse")?.value);
  const bmiEl = json._mode === "awam"
    ? parseFloat(document.getElementById("aw_bmi_display")?.value)
    : parseFloat(document.getElementById("bmi")?.value);

  if (!isNaN(glucoseEl) && glucoseEl > 100) items.push("Kadar glukosa di atas normal — disarankan pemantauan kadar gula darah secara berkala.");
  if (!isNaN(pulseEl)   && pulseEl > 100)   items.push("Detak jantung di atas normal — disarankan pemeriksaan EKG lebih lanjut.");
  if (!isNaN(bmiEl)     && bmiEl > 25)      items.push("BMI menunjukkan kelebihan berat badan — disarankan konsultasi nutrisi.");
  if (items.length === 0) items.push("Kondisi klinis dalam batas normal berdasarkan nilai yang dimasukkan.");

  box.innerHTML = items.map(i => `<li>${i}</li>`).join("");
}

// ============================================================
// RESET FORM — MODE TENAGA MEDIS
// ============================================================
const resetBtn = document.getElementById("resetBtn");
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    document.getElementById("predictForm")?.reset();
    const sec = document.getElementById("resultSection");
    if (sec) sec.style.display = "none";
    document.querySelectorAll(".is-invalid").forEach(el => el.classList.remove("is-invalid"));
  });
}

// ============================================================
// DATA SAMPLE
// ============================================================
const sampleBtn = document.getElementById("sampleBtn");
if (sampleBtn) {
  sampleBtn.addEventListener("click", async () => {
    showLoading();
    try {
      const res  = await fetch("/api/sample");
      const json = await res.json();
      hideLoading();
      if (json.status !== "ok") return alert(json.message);

      const d      = json.data;
      const set    = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
      const setChk = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };

      set("rcount",         d.rcount);
      set("gender",         d.gender);
      set("facid",          d.facid);
      set("hematocrit",     d.hematocrit);
      set("neutrophils",    d.neutrophils);
      set("sodium",         d.sodium);
      set("glucose",        d.glucose);
      set("bloodureanitro", d.bloodureanitro);
      set("creatinine",     d.creatinine);
      set("bmi",            d.bmi);
      set("pulse",          d.pulse);
      set("respiration",    d.respiration);

      setChk("dialysisrenalendstage",      d.dialysisrenalendstage);
      setChk("asthma",                     d.asthma);
      setChk("irondef",                    d.irondef);
      setChk("pneum",                      d.pneum);
      setChk("substancedependence",        d.substancedependence);
      setChk("psychologicaldisordermajor", d.psychologicaldisordermajor);
      setChk("depress",                    d.depress);
      setChk("psychother",                 d.psychother);
      setChk("fibrosisandother",           d.fibrosisandother);
      setChk("malnutrition",               d.malnutrition);
      setChk("hemo",                       d.hemo);
      setChk("secondarydiagnosisnonicd9",  d.secondarydiagnosisnonicd9);

      const note = document.getElementById("sampleNote");
      if (note) note.textContent = `Data acak dimuat. Nilai aktual lama rawat: ${d.actual_los} hari.`;

      document.getElementById("tab-manual-btn")?.click();
      document.getElementById("predictForm")?.scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      hideLoading();
      alert("Gagal mengambil data sampel.");
    }
  });
}

// ============================================================
// UPLOAD CSV
// ============================================================
const uploadZone = document.getElementById("uploadZone");
const fileInput  = document.getElementById("csvFile");

if (uploadZone && fileInput) {
  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", e => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));

  uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) handleCSVFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleCSVFile(fileInput.files[0]);
  });
}

function handleCSVFile(file) {
  const label = document.getElementById("uploadLabel");
  if (label) label.textContent = `File dipilih: ${file.name}`;

  const reader = new FileReader();
  reader.onload = (e) => {
    const lines   = e.target.result.split("\n").slice(0, 6);
    const headers = lines[0].split(",");
    let html = `<div class="table-responsive mt-3"><table class="table table-sm">
      <thead><tr>${headers.map(h => `<th>${h.trim()}</th>`).join("")}</tr></thead><tbody>`;
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      html += `<tr>${lines[i].split(",").map(c => `<td>${c.trim()}</td>`).join("")}</tr>`;
    }
    html += "</tbody></table></div>";
    const preview = document.getElementById("csvPreview");
    if (preview) preview.innerHTML = html;
  };
  reader.readAsText(file);

  const submitBtn = document.getElementById("csvSubmitBtn");
  if (submitBtn) submitBtn.style.display = "block";
}

const csvSubmitBtn = document.getElementById("csvSubmitBtn");
if (csvSubmitBtn) {
  csvSubmitBtn.addEventListener("click", async () => {
    const file = fileInput?.files[0];
    if (!file) return alert("Pilih file CSV terlebih dahulu.");

    const formData = new FormData();
    formData.append("file", file);
    showLoading();
    try {
      const res = await fetch("/api/predict-csv", { method: "POST", body: formData });
      hideLoading();
      if (res.ok) {
        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement("a");
        a.href = url; a.download = "hasil_prediksi.csv"; a.click();
      } else {
        const json = await res.json();
        alert("Error: " + json.message);
      }
    } catch (e) {
      hideLoading();
      alert("Gagal memproses file.");
    }
  });
}

// ============================================================
// MODE PASIEN — HITUNG BMI OTOMATIS
// ============================================================
const awWeight     = document.getElementById("aw_weight");
const awHeight     = document.getElementById("aw_height");
const awBmiDisplay = document.getElementById("aw_bmi_display");

function hitungBMI() {
  const bb = parseFloat(awWeight?.value);
  const tb = parseFloat(awHeight?.value) / 100;
  if (!isNaN(bb) && !isNaN(tb) && tb > 0) {
    const bmi = (bb / (tb * tb)).toFixed(1);
    if (awBmiDisplay) awBmiDisplay.value = bmi;
  }
}

if (awWeight) awWeight.addEventListener("input", hitungBMI);
if (awHeight) awHeight.addEventListener("input", hitungBMI);

// ============================================================
// MODE PASIEN — VALIDASI
// ============================================================
function validateAwamForm() {
  let valid = true;
  const fields = [
    { id: "aw_weight", label: "Berat badan" },
    { id: "aw_height", label: "Tinggi badan" },
    { id: "aw_pulse",  label: "Detak jantung" }
  ];
  fields.forEach(f => {
    const el  = document.getElementById(f.id);
    const err = document.getElementById(f.id + "_err");
    if (!el) return;
    el.classList.remove("is-invalid");
    if (!el.value.trim() || isNaN(parseFloat(el.value))) {
      el.classList.add("is-invalid");
      if (err) err.textContent = f.label + " wajib diisi.";
      valid = false;
    }
  });
  return valid;
}

// ============================================================
// MODE PASIEN — PETAKAN JAWABAN KE NILAI LAB
// ============================================================
function buildAwamData() {
  const bb     = parseFloat(document.getElementById("aw_weight")?.value);
  const tb     = parseFloat(document.getElementById("aw_height")?.value) / 100;
  const bmi    = parseFloat((bb / (tb * tb)).toFixed(2));
  const kidney = document.getElementById("aw_kidney")?.value || "normal";

  const creatinine_map = { normal: 0.9,  mild: 1.5, moderate: 2.5, severe: 4.0 };
  const bun_map        = { normal: 14,   mild: 25,  moderate: 40,  severe: 60  };

  return {
    rcount:  document.getElementById("aw_rcount")?.value  || "0",
    gender:  document.getElementById("aw_gender")?.value  || "F",
    facid:   document.getElementById("aw_facid")?.value   || "A",
    bmi:     bmi,
    pulse:   parseInt(document.getElementById("aw_pulse")?.value) || 80,
    glucose:        parseFloat(document.getElementById("aw_glucose")?.value)     || 90,
    respiration:    parseFloat(document.getElementById("aw_respiration")?.value) || 16,
    hematocrit:     parseFloat(document.getElementById("aw_anemia")?.value)      || 42,
    neutrophils:    parseFloat(document.getElementById("aw_infection")?.value)   || 62,
    sodium:         parseFloat(document.getElementById("aw_hydration")?.value)   || 139,
    creatinine:     creatinine_map[kidney] || 0.9,
    bloodureanitro: bun_map[kidney]        || 14,
    dialysisrenalendstage:      document.getElementById("aw_dialysis")?.checked   ? 1 : 0,
    asthma:                     document.getElementById("aw_asthma")?.checked     ? 1 : 0,
    irondef:                    document.getElementById("aw_irondef")?.checked    ? 1 : 0,
    pneum:                      document.getElementById("aw_pneum")?.checked      ? 1 : 0,
    substancedependence:        document.getElementById("aw_substance")?.checked  ? 1 : 0,
    psychologicaldisordermajor: document.getElementById("aw_psych")?.checked      ? 1 : 0,
    depress:                    document.getElementById("aw_depress")?.checked    ? 1 : 0,
    psychother:                 document.getElementById("aw_psychother")?.checked ? 1 : 0,
    fibrosisandother:           document.getElementById("aw_fibrosis")?.checked   ? 1 : 0,
    malnutrition:               document.getElementById("aw_malnut")?.checked     ? 1 : 0,
    hemo:                       document.getElementById("aw_hemo")?.checked       ? 1 : 0,
    secondarydiagnosisnonicd9:  document.getElementById("aw_secondary")?.checked  ? 1 : 0,
  };
}

// ============================================================
// MODE PASIEN — TOMBOL PREDIKSI
// ============================================================
const awamPredictBtn = document.getElementById("awamPredictBtn");
if (awamPredictBtn) {
  awamPredictBtn.addEventListener("click", async () => {
    if (!validateAwamForm()) return;

    const data = buildAwamData();
    showLoading();
    try {
      const res  = await fetch("/api/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(data)
      });
      const json = await res.json();
      hideLoading();
      if (json.status === "ok") {
        json._mode = "awam";
        renderResult(json);
      } else {
        alert("Terjadi kesalahan: " + json.message);
      }
    } catch (e) {
      hideLoading();
      alert("Gagal menghubungi server.");
    }
  });
}

// ============================================================
// MODE PASIEN — TOMBOL RESET
// ============================================================
const awamResetBtn = document.getElementById("awamResetBtn");
if (awamResetBtn) {
  awamResetBtn.addEventListener("click", () => {
    document.getElementById("awamForm")?.reset();
    if (awBmiDisplay) awBmiDisplay.value = "";
    const sec = document.getElementById("resultSection");
    if (sec) sec.style.display = "none";
    document.querySelectorAll(".is-invalid").forEach(el => el.classList.remove("is-invalid"));
  });
}

// ============================================================
// CHART.JS — COMPARE PAGE
// ============================================================
function buildCompareCharts(metrics, annLoss, annVal, lstmLoss, lstmVal, elbowK, elbowIn) {
  const modelNames = ["Linear Regression", "ANN", "LSTM", "Backpropagation"];
  const modelKeys  = ["linear", "ann", "lstm", "backprop"];
  const colors     = ["#93C5FD", "#2563EB", "#1D4ED8", "#60A5FA"];

  const ctxRmse = document.getElementById("chartRMSE");
  if (ctxRmse && metrics) {
    new Chart(ctxRmse, {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [{
          label: "RMSE",
          data: modelKeys.map(k => metrics[k]?.RMSE || 0),
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: "rgba(0,0,0,.05)" } } },
        animation: { duration: 800 }
      }
    });
  }

  const ctxR2 = document.getElementById("chartR2");
  if (ctxR2 && metrics) {
    new Chart(ctxR2, {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [{
          label: "R² Score",
          data: modelKeys.map(k => metrics[k]?.R2 || 0),
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 1, grid: { color: "rgba(0,0,0,.05)" } } },
        animation: { duration: 800 }
      }
    });
  }

  const ctxAnn = document.getElementById("chartANNLoss");
  if (ctxAnn && annLoss.length) {
    new Chart(ctxAnn, {
      type: "line",
      data: {
        labels: annLoss.map((_, i) => i + 1),
        datasets: [
          { label: "Train Loss", data: annLoss, borderColor: "#2563EB",
            tension: .3, pointRadius: 0, borderWidth: 2 },
          { label: "Val Loss",   data: annVal,  borderColor: "#EF4444",
            tension: .3, pointRadius: 0, borderWidth: 2, borderDash: [5,5] }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: { x: { ticks: { maxTicksLimit: 10 } } },
        animation: { duration: 800 }
      }
    });
  }

  const ctxLstm = document.getElementById("chartLSTMLoss");
  if (ctxLstm && lstmLoss.length) {
    new Chart(ctxLstm, {
      type: "line",
      data: {
        labels: lstmLoss.map((_, i) => i + 1),
        datasets: [
          { label: "Train Loss", data: lstmLoss, borderColor: "#2563EB",
            tension: .3, pointRadius: 0, borderWidth: 2 },
          { label: "Val Loss",   data: lstmVal,  borderColor: "#EF4444",
            tension: .3, pointRadius: 0, borderWidth: 2, borderDash: [5,5] }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: { x: { ticks: { maxTicksLimit: 10 } } },
        animation: { duration: 800 }
      }
    });
  }

  const ctxElbow = document.getElementById("chartElbow");
  if (ctxElbow && elbowK.length) {
    new Chart(ctxElbow, {
      type: "line",
      data: {
        labels: elbowK,
        datasets: [{
          label: "Inertia",
          data: elbowIn,
          borderColor: "#2563EB",
          backgroundColor: "rgba(37,99,235,.08)",
          fill: true,
          tension: .3,
          pointRadius: 5,
          pointBackgroundColor: elbowK.map(k => k === 3 ? "#EF4444" : "#2563EB")
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "Jumlah Cluster (k)" } } },
        animation: { duration: 800 }
      }
    });
  }
}