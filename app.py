BUILD_ID = "RENDER_PROOF_123"

from __future__ import annotations

# ---------- Imports ----------
import os, json, math
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# LightGBM
try:
    import lightgbm as lgb
except Exception:
    lgb = None

BUILD_ID = "BUILD_2025_12_21_A"

# =========================
# 1) Health & Root
# =========================
app = FastAPI(title="ProcureSight API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://procuresight.streamlit.app",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check() -> dict:
    return {"ok": True, "build": BUILD_ID, "file": __file__, "cwd": os.getcwd()}

@app.get("/")
def root() -> dict:
    try:
        _ensure_loaded()
        ok = bool(_clf and _reg_s and _reg_l and _features)
        return {"ok": True, "model_loaded": ok, "name": "ProcureSight API", "build": BUILD_ID, "file": __file__}
    except Exception as e:
        return {"ok": False, "model_loaded": False, "error": str(e), "build": BUILD_ID, "file": __file__}

@app.get("/where")
def where() -> dict:
    return {"ok": True, "build": BUILD_ID, "file": __file__, "cwd": os.getcwd()}


# =========================
# 2) Schemas
# =========================
class PredictRequest(BaseModel):
    tender_country: Optional[str] = None
    tender_mainCpv: Optional[str] = None
    tender_year: Optional[int] = None
    tender_procedureType: Optional[str] = None
    tender_supplyType: Optional[str] = None
    buyer_buyerType: Optional[str] = None
    buyer_country: Optional[str] = None
    tender_estimatedPrice_EUR: Optional[float] = None
    tender_indicator_score_INTEGRITY: Optional[float] = None
    tender_indicator_score_ADMINISTRATIVE: Optional[float] = None
    tender_indicator_score_TRANSPARENCY: Optional[float] = None
    lot_bidsCount: Optional[float] = None
    tender_estimatedPrice_EUR_log: Optional[float] = None
    lot_bidsCount_log: Optional[float] = None
    target_duration: Optional[float] = None

class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_days: float
    risk_flag: bool
    model_used: str = "lgbm_2stage"

    # τ σε ημέρες (UI threshold)
    tau_days: float

    # internals
    p_long: float
    tau_prob: float
    stage_used: str
    pred_short: float
    pred_long: float

    # debug
    build: str


# =========================
# 3) Helpers  (+ Loader εδώ)
# =========================
# globals loaded once
_clf = None
_reg_s = None
_reg_l = None
_features: Dict[str, Any] = {}
_meta: Dict[str, Any] = {}
_LONG_THR_DEFAULT = 720.0

def _ensure_loaded() -> None:
    global _clf, _reg_s, _reg_l, _features, _meta, _LONG_THR_DEFAULT
    if _clf is not None and _reg_s is not None and _reg_l is not None and _features:
        return

    if lgb is None:
        raise RuntimeError("LightGBM is not installed (missing lightgbm in requirements).")

    model_dir = Path(__file__).resolve().parent
    if not model_dir.exists():
        raise RuntimeError(f"Model dir not found: {model_dir}")

    _clf   = lgb.Booster(model_file=str(model_dir / "stage1_classifier.txt"))
    _reg_s = lgb.Booster(model_file=str(model_dir / "stage2_reg_short.txt"))
    _reg_l = lgb.Booster(model_file=str(model_dir / "stage2_reg_long.txt"))

    with open(model_dir / "features.json", "r", encoding="utf-8") as f:
        _features = json.load(f)

    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        _LONG_THR_DEFAULT = float(_meta.get("long_threshold_days", _LONG_THR_DEFAULT))

def _align_to_booster(X: pd.DataFrame, booster) -> pd.DataFrame:
    try:
        exp = list(booster.feature_name())
    except Exception:
        exp = None
    if not exp or any((n is None) or (n == "") for n in exp):
        return X
    for c in exp:
        if c not in X.columns:
            X[c] = np.nan
    return X[exp]

def _derive_cpv_parts(s: pd.Series):
    s = s.astype(str).str.replace(r"[^\d]", "", regex=True).str.zfill(8)
    return pd.to_numeric(s.str[:2], errors="coerce"), pd.to_numeric(s.str[:3], errors="coerce")

def _build_X(req: PredictRequest) -> pd.DataFrame:
    d = req.model_dump()

    d["tender_country"] = (d.get("tender_country") or "").upper().strip()
    if d.get("tender_mainCpv") is not None:
        d["tender_mainCpv"] = str(d["tender_mainCpv"]).strip()

    X = pd.DataFrame([d])

    if "tender_mainCpv" in X.columns and X["tender_mainCpv"].notna().any():
        div2, grp3 = _derive_cpv_parts(X["tender_mainCpv"])
        X["cpv_div2"], X["cpv_grp3"] = div2, grp3

    feat_list = list(_features.get("features", []))
    cat_list = set(_features.get("categorical", []))

    for c in feat_list:
        if c not in X.columns:
            X[c] = np.nan
    if feat_list:
        X = X[feat_list]

    # auto log1p
    if "tender_estimatedPrice_EUR" in X.columns and "tender_estimatedPrice_EUR_log" in X.columns:
        base = pd.to_numeric(X["tender_estimatedPrice_EUR"], errors="coerce")
        X["tender_estimatedPrice_EUR_log"] = np.log1p(base)

    if "lot_bidsCount" in X.columns and "lot_bidsCount_log" in X.columns:
        base = pd.to_numeric(X["lot_bidsCount"], errors="coerce")
        X["lot_bidsCount_log"] = np.log1p(base)

    if "target_duration" in X.columns and X["target_duration"].isna().any():
        X["target_duration"] = 700.0

    # dtypes
    for c in X.columns:
        if c in cat_list:
            X[c] = X[c].astype("string").str.strip().astype("category")
        else:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

    if "tender_year" in X.columns:
        X["tender_year"] = pd.to_numeric(X["tender_year"], errors="coerce")

    return X

def _combine_hard(p_long: float, y_short: float, y_long: float, tau_prob: float) -> float:
    out = y_long if (p_long >= tau_prob) else y_short
    return float(np.clip(out, 1.0, 1800.0))

def _year_bump(year: Optional[int]) -> float:
    if year is None or not _meta:
        return 0.0
    bump = (_meta.get("bump") or {})
    if bump.get("mode") != "year-aware":
        return 0.0
    info = _meta.get("year_bump_info") or {}
    coeffs = info.get("poly_coeffs") or []
    if not coeffs:
        return 0.0
    max_b = float(info.get("max_b", bump.get("year_bump_max", 5.0)))
    y = float(year)
    val = 0.0
    for i, a in enumerate(coeffs):
        val += float(a) * (y ** i)
    return float(np.clip(val, 0.0, max_b))

def _apply_year_bump(yhat: float, year: Optional[int]) -> float:
    b = _year_bump(year)
    LOW, HIGH, CENTER, SNAP_FROM, SNAP_TO = 670.0, 720.0, 720.0, 715.0, 720.0
    if yhat < LOW or yhat >= HIGH or b <= 0:
        return float(yhat)
    w = (yhat - LOW) / max(CENTER - LOW, 1e-9)
    y2 = yhat + b * max(min(w, 1.0), 0.0)
    if SNAP_FROM <= y2 < SNAP_TO:
        y2 = SNAP_TO
    return float(y2)


# =========================
# 4) Predict
# =========================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, tau: Optional[float] = Query(None, description="threshold (days)")):
    try:
        _ensure_loaded()

        tau_days = float(tau) if tau is not None else float(_LONG_THR_DEFAULT)
        tau_prob = float(_meta.get("tau", 0.5))

        X = _build_X(req)
        Xc = _align_to_booster(X.copy(), _clf)
        Xs = _align_to_booster(X.copy(), _reg_s)
        Xl = _align_to_booster(X.copy(), _reg_l)

        p = float(_clf.predict(Xc)[0])
        if (p is None) or (not math.isfinite(p)):
            p = 0.0

        y_short = float(_reg_s.predict(Xs)[0])
        y_long  = float(_reg_l.predict(Xl)[0])

        yhat = _combine_hard(p, y_short, y_long, tau_prob)
        yhat = _apply_year_bump(yhat, req.tender_year)

        stage_used = "long_reg" if (p >= tau_prob) else "short_reg"
        risk_flag = bool(yhat >= tau_days)

        return PredictResponse(
            predicted_days=float(yhat),
            risk_flag=risk_flag,
            tau_days=float(tau_days),
            p_long=float(p),
            tau_prob=float(tau_prob),
            stage_used=stage_used,
            pred_short=float(y_short),
            pred_long=float(y_long),
            build=BUILD_ID,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
