from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import shap
import os

# ── Paths ──
BASE_PATH  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "models")
DATA_PATH  = os.path.join(BASE_PATH, "data", "processed")

# ── Feature names ──
FEATURE_NAMES = [
    "Age_of_the_patient",
    "Gender_of_the_patient",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkphos_Alkaline_Phosphotase",
    "Sgpt_Alamine_Aminotransferase",
    "Sgot_Aspartate_Aminotransferase",
    "Total_Protiens",
    "ALB_Albumin",
    "A_G_Ratio_Albumin_and_Globulin_Ratio"
]
N_FEATURES = len(FEATURE_NAMES)   # 10

# ── HepaXAI Architecture ──
class HepaXAI(nn.Module):
    def __init__(self, clin_dim, lab_dim, bio_dim, hidden=128, dropout=0.2):
        super(HepaXAI, self).__init__()
        self.stream_clinical = nn.Sequential(
            nn.Linear(clin_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.stream_lab = nn.Sequential(
            nn.Linear(lab_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.encoder = nn.Sequential(
            nn.Linear(bio_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, bio_dim))
        fusion_dim = (hidden // 2) * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.head_risk     = nn.Linear(hidden // 2, 1)
        self.head_stage    = nn.Linear(hidden // 2, 4)
        self.head_severity = nn.Linear(hidden // 2, 3)

    def forward(self, x_clin, x_lab, x_bio):
        out_clin = self.stream_clinical(x_clin)
        out_lab  = self.stream_lab(x_lab)
        out_bio  = self.encoder(x_bio)
        fused    = torch.cat([out_clin, out_lab, out_bio], dim=1)
        fused    = self.fusion(fused)
        risk     = torch.sigmoid(self.head_risk(fused)).squeeze(1)
        stage    = self.head_stage(fused)
        severity = self.head_severity(fused)
        return risk, stage, severity


# ── Global state ──
device = torch.device("cpu")
model           = None
scaler          = None
background_data = None


# ── Lifespan (replaces deprecated on_event) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, background_data

    # ── Load scaler ──
    scaler_path = os.path.join(DATA_PATH, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded!")

    # ✅ FIX — tous les streams reçoivent 10 features (comme à l'entraînement)
    model_instance = HepaXAI(
        clin_dim = N_FEATURES,   # 10  ← était len(CLINICAL_FEATURES) = 2
        lab_dim  = N_FEATURES,   # 10  ← était len(LAB_FEATURES) = 8
        bio_dim  = N_FEATURES,   # 10  ✅ déjà correct
    )
    model_path = os.path.join(MODEL_PATH, "hepaxai_best.pt")
    model_instance.load_state_dict(
        torch.load(model_path, map_location=device))
    model_instance.eval()
    model = model_instance
    print("✅ HepaXAI model loaded!")

    # ── Background data for SHAP ──
    X_train_path = os.path.join(DATA_PATH, "X_train.csv")
    if os.path.exists(X_train_path):
        X_train = pd.read_csv(X_train_path)
        background_data = X_train.sample(50, random_state=42).values
        print("✅ Background data loaded for SHAP!")

    yield   # ← app runs here

    # shutdown (rien à faire)
    print("🛑 HepaXAI API shutting down.")


# ── App ──
app = FastAPI(title="HepaXAI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input schema ──
class PatientData(BaseModel):
    age:                  float
    gender:               int     # 0=Male, 1=Female
    total_bilirubin:      float
    direct_bilirubin:     float
    alkaline_phosphotase: float
    alt:                  float
    ast:                  float
    total_proteins:       float
    albumin:              float
    ag_ratio:             float


# ── SHAP wrapper ─────────────────────────────────────────────────────────
def predict_proba(X_array: np.ndarray) -> np.ndarray:
    """Wrapper used by shap.KernelExplainer — all 3 streams get full 10 features."""
    x = torch.tensor(X_array.astype(np.float32))
    with torch.no_grad():
        risk, _, _ = model(x, x, x)   # ✅ same tensor to every stream
    probs = risk.numpy()
    return np.column_stack([1 - probs, probs])


# ── /predict ─────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(patient: PatientData):
    # 1. Build raw feature array (order must match FEATURE_NAMES)
    raw = np.array([[
        patient.age,
        patient.gender,
        patient.total_bilirubin,
        patient.direct_bilirubin,
        patient.alkaline_phosphotase,
        patient.alt,
        patient.ast,
        patient.total_proteins,
        patient.albumin,
        patient.ag_ratio,
    ]])

    # 2. Scale
    scaled = scaler.transform(raw)          # shape (1, 10)
    x      = torch.tensor(scaled.astype(np.float32))

    # 3. Predict — ✅ tous les streams reçoivent les 10 features
    with torch.no_grad():
        risk, stage, severity = model(x, x, x)

    risk_prob     = float(risk[0].item())
    stage_pred    = int(torch.argmax(stage[0]).item())
    severity_pred = int(torch.argmax(severity[0]).item())

    stage_map    = {0: "Stage I", 1: "Stage II", 2: "Stage III", 3: "Stage IV"}
    severity_map = {0: "Mild", 1: "Moderate", 2: "Severe"}

    # 4. SHAP
    shap_values_list = []
    if background_data is not None:
        try:
            explainer = shap.KernelExplainer(predict_proba, background_data)
            shap_vals = explainer.shap_values(scaled, nsamples=50)
            # shap_vals is list[2] for binary → take class-1 slice
            shap_class1 = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            shap_values_list = sorted(
                [{"feature": FEATURE_NAMES[i], "value": float(shap_class1[i])}
                 for i in range(N_FEATURES)],
                key=lambda x: abs(x["value"]), reverse=True
            )
        except Exception as e:
            print(f"⚠️  SHAP error: {e}")

    return {
        "risk_score":  round(risk_prob * 100, 2),
        "risk_label":  "High Risk" if risk_prob > 0.5 else "Low Risk",
        "stage":       stage_map[stage_pred],
        "severity":    severity_map[severity_pred],
        "confidence":  round(max(risk_prob, 1 - risk_prob) * 100, 2),
        "shap_values": shap_values_list,
    }


# ── /health ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  model is not None,
        "scaler_loaded": scaler is not None,
    }


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)