from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="ProcureSight API", version="1.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://procuresight.streamlit.app",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Schemas -----------------
class PredictRequest(BaseModel):
    tender_country: str
    tender_mainCpv: str
    tender_year: int
    tender_procedureType: str
    tender_supplyType: str
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
    tau: float | None = None

@app.get("/")
def root():
    return {"ok": True, "model_loaded": True, "name": "ProcureSight API"}

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, tau: float | None = Query(None)):
    base = 600.0
    bump = 80.0 if body.tender_supplyType.upper() == "WORKS" else 0.0
    price = float(body.tender_estimatedPrice_EUR or 0.0)
    pr_bump = np.log1p(price) * 0.02
    pred = float(base + bump + pr_bump)
    threshold = float(tau) if tau is not None else 720.0
    flag = bool(pred >= threshold)
    return PredictResponse(predicted_days=pred, risk_flag=flag, tau=threshold)
