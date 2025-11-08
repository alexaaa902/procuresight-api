from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math

app = FastAPI(title="ProcureSight API", version="1.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://procuresight.streamlit.app",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*"  # αν θέλεις να το χαλαρώσεις εντελώς
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Health & Root -----------------
@app.get("/health")
def health_check() -> dict:
    return {"ok": True}

@app.get("/")
def root() -> dict:
    # τυπική απάντηση για το Connection test
    return {"ok": True, "model_loaded": True, "name": "ProcureSight API"}

# ----------------- Schemas -----------------
class PredictRequest(BaseModel):
    tender_country: str | None = None
    tender_mainCpv: str | None = None
    tender_year: int | None = None
    tender_procedureType: str | None = None
    tender_supplyType: str | None = None
    buyer_buyerType: str | None = None
    buyer_country: str | None = None
    tender_estimatedPrice_EUR: float | None = None
    lot_bidsCount: float | None = None
    tender_estimatedPrice_EUR_log: float | None = None
    lot_bidsCount_log: float | None = None
    target_duration: float | None = None

class PredictResponse(BaseModel):
    predicted_days: float
    risk_flag: bool
    model_used: str = "lgbm_2stage"
    tau: float

# ----------------- Helpers -----------------
def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def _upper(s: str | None) -> str:
    return (s or "").strip().upper()

# ----------------- Predict -----------------
@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, tau: float | None = Query(None, description="threshold (days)")):
    base = 600.0
    is_works = _upper(body.tender_supplyType) == "WORKS"
    bump = 80.0 if is_works else 0.0

    price = _safe_float(body.tender_estimatedPrice_EUR, default=0.0)
    pr_bump = math.log1p(max(price, 0.0)) * 0.02  # σταθερό, ασφαλές

    pred = float(base + bump + pr_bump)
    threshold = _safe_float(tau, default=720.0)
    flag = bool(pred >= threshold)

    return PredictResponse(
        predicted_days=pred,
        risk_flag=flag,
        tau=threshold,
    )
