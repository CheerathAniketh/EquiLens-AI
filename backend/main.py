import os
import threading
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional
from fastapi.responses import Response
from pdf_exporter import generate_audit_pdf

from analyzer import analyze_bias, compute_intersectionality, compute_eod
from trainer import train_and_evaluate, prepare_features
from explainer import get_shap_values
from gemini_client import explain_results, suggest_fixes, explain_whatif

app = FastAPI(title="EquiLens AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*", "X-Session-Id"],
)

# ─── Multi-user session cache ──────────────────────────────────────────────────
# Each browser gets a unique session ID so concurrent judges don't clash.
session_cache: dict = {}
session_lock = threading.Lock()
MAX_SESSIONS = 100


def _evict_oldest():
    if len(session_cache) >= MAX_SESSIONS:
        oldest = next(iter(session_cache))
        del session_cache[oldest]


# ─── Pydantic models ──────────────────────────────────────────────────────────

class WhatIfRequest(BaseModel):
    drop_features: List[str]
    audience: str = "ngo"


# ─── /analyze ─────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_col: str = "",
    sensitive_col: str = "",
    sensitive_col_2: str = "",
    audience: str = "ngo",
    x_session_id: Optional[str] = Header(default=None),
):
    try:
        df = pd.read_csv(file.file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise HTTPException(
            status_code=400,
            detail="Could not parse the uploaded file. Please upload a valid, non-empty CSV."
        )

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{target_col}' not found in the uploaded CSV."
        )

    if sensitive_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{sensitive_col}' not found in the uploaded CSV."
        )

    with session_lock:
        if x_session_id and x_session_id in session_cache:
            session_id = x_session_id
        else:
            session_id = str(uuid.uuid4())
            while session_id in session_cache:
                session_id = str(uuid.uuid4())

        _evict_oldest()

        session_cache[session_id] = {
            "df": df,
            "target_col": target_col,
            "sensitive_col": sensitive_col,
        }

    stats = analyze_bias(df, target_col, sensitive_col)
    model, X_train, X_test, y_test, curves, y_pred, y_prob, sensitive_test = train_and_evaluate(
        df, target_col, sensitive_col
    )
    _, y_all = prepare_features(df, target_col, sensitive_col)
    y_train = y_all.loc[X_train.index]

    with session_lock:
        if session_id in session_cache:
            session_cache[session_id].update({
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            })

    shap_data = get_shap_values(model, X_train, X_test)

    eod_data = compute_eod(y_test, y_pred, sensitive_test)
    stats["eod"] = eod_data["eod"]
    stats["eod_details"] = eod_data

    with session_lock:
        if session_id in session_cache:
            session_cache[session_id]["original_eod"] = eod_data["eod"]

    intersectionality = None
    if sensitive_col_2 and sensitive_col_2 != sensitive_col and sensitive_col_2 in df.columns:
        intersectionality = compute_intersectionality(df, target_col, sensitive_col, sensitive_col_2)

    try:
        explanation = explain_results(stats, shap_data, audience=audience)
    except Exception as e:
        explanation = f"Gemini unavailable: {str(e)}"

    try:
        fixes = suggest_fixes(stats, shap_data)
    except Exception as e:
        fixes = [
            "Rebalance your dataset so all groups have equal representation.",
            "Remove proxy features that correlate with the sensitive attribute.",
            "Collect more representative data from underrepresented groups.",
        ]

    return {
        "session_id": session_id,
        "stats": stats,
        "shap": shap_data,
        "explanation": explanation,
        "fixes": fixes,
        "curves": curves,
        "intersectionality": intersectionality,
    }


# ─── /whatif/features ─────────────────────────────────────────────────────────

@app.get("/whatif/features")
async def whatif_features(x_session_id: Optional[str] = Header(default=None)):
    with session_lock:
        if not x_session_id or x_session_id not in session_cache:
            raise HTTPException(
                status_code=400,
                detail="No dataset in cache. Run /analyze first."
            )
        session = session_cache[x_session_id]

    df = session["df"]
    target_col = session["target_col"]
    sensitive_col = session["sensitive_col"]
    features = [c for c in df.columns if c not in [target_col, sensitive_col]]
    return {"features": features}


# ─── /whatif ──────────────────────────────────────────────────────────────────

@app.post("/whatif")
async def whatif(
    body: WhatIfRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    with session_lock:
        if not x_session_id or x_session_id not in session_cache:
            raise HTTPException(
                status_code=400,
                detail="No dataset in cache. Run /analyze first to upload a CSV."
            )

        session = session_cache[x_session_id]

    df = session["df"]
    target_col = session["target_col"]
    sensitive_col = session["sensitive_col"]
    X_train_cached = session.get("X_train")
    X_test_cached = session.get("X_test")
    y_train_cached = session.get("y_train")
    y_test_cached = session.get("y_test")
    original_eod_cached = session.get("original_eod")

    if any(v is None for v in [X_train_cached, X_test_cached, y_train_cached, y_test_cached]):
        raise HTTPException(
            status_code=400,
            detail="No cached train/test split found. Run /analyze first to upload a CSV."
        )

    all_features = [c for c in df.columns if c != target_col]
    invalid = [f for f in body.drop_features if f not in all_features]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Features not found in dataset: {invalid}. Available: {all_features}"
        )

    # ── Original metrics ───────────────────────────────────────────────────────
    original_stats = analyze_bias(df, target_col, sensitive_col)
    original_stats["eod"] = float(original_eod_cached) if original_eod_cached is not None else 0.0

    # ── Modified: retrain on reduced feature set ───────────────────────────────
    df_modified = df.drop(columns=body.drop_features, errors="ignore")

    X_train_mod = X_train_cached.drop(columns=body.drop_features, errors="ignore")
    X_test_mod = X_test_cached.drop(columns=body.drop_features, errors="ignore")

    model_mod = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model_mod.fit(X_train_mod, y_train_cached)

    y_pred_mod = model_mod.predict(X_test_mod)
    sensitive_test_mod = df.loc[y_test_cached.index, sensitive_col].astype(str).reset_index(drop=True)

    pred_df = pd.DataFrame({
        sensitive_col: sensitive_test_mod.values,
        target_col: y_pred_mod,
    })
    modified_stats = analyze_bias(pred_df, target_col, sensitive_col)

    eod_mod = compute_eod(y_test_cached, y_pred_mod, sensitive_test_mod)
    modified_stats["eod"] = eod_mod["eod"]

    shap_modified = get_shap_values(model_mod, X_train_mod, X_test_mod)

    delta = {
        "spd": round(original_stats["spd"] - modified_stats["spd"], 4),
        "di":  round(modified_stats["di"]  - original_stats["di"],  4),
        "eod": round(original_stats["eod"] - modified_stats["eod"], 4),
    }

    try:
        whatif_explanation = explain_whatif(
            original=original_stats,
            modified=modified_stats,
            delta=delta,
            dropped_features=body.drop_features,
            audience=body.audience,
        )
    except Exception:
        whatif_explanation = _fallback_whatif_explanation(
            original_stats, modified_stats, delta, body.drop_features
        )

    return {
        "original": {
            "spd":         original_stats["spd"],
            "di":          original_stats["di"],
            "eod":         original_stats["eod"],
            "severity":    original_stats["severity"],
            "group_stats": original_stats["group_stats"],
        },
        "modified": {
            "spd":         modified_stats["spd"],
            "di":          modified_stats["di"],
            "eod":         modified_stats["eod"],
            "severity":    modified_stats["severity"],
            "group_stats": modified_stats["group_stats"],
        },
        "delta":              delta,
        "dropped_features":   body.drop_features,
        "remaining_features": [
            c for c in df_modified.columns
            if c not in [target_col, sensitive_col]
        ],
        "shap":        shap_modified,
        "explanation": whatif_explanation,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fallback_whatif_explanation(original, modified, delta, dropped):
    spd_dir = "decreased" if delta["spd"] > 0 else "increased"
    di_dir  = "improved"  if delta["di"]  > 0 else "worsened"
    eod_dir = "decreased" if delta["eod"] > 0 else "increased"
    dropped_str = ", ".join(dropped)

    threshold_note = ""
    if original["di"] < 0.8 and modified["di"] >= 0.8:
        threshold_note = " The model has now crossed the legal fairness threshold of 0.80 — this is a meaningful improvement."
    elif modified["di"] < 0.8:
        threshold_note = (
            f" The Disparate Impact is still below the legal threshold of 0.80 "
            f"(currently {modified['di']:.3f}), so further action is needed."
        )

    outcome = (
        "These features appear to have been contributing to bias — consider permanently removing them."
        if delta["di"] > 0
        else "Removing these features did not reduce bias. The discrimination likely comes from other features — try dropping different columns or rebalancing the dataset."
    )

    return (
        f"After dropping {dropped_str}, the outcome gap between groups {spd_dir} "
        f"by {abs(delta['spd']):.3f} points. "
        f"The Disparate Impact ratio {di_dir} from {original['di']:.3f} to {modified['di']:.3f}."
        f"{threshold_note} "
        f"Equalized Odds {eod_dir} from {original['eod']:.3f} to {modified['eod']:.3f}.\n\n"
        f"{outcome} *"
    )


# ─── PDF export ───────────────────────────────────────────────────────────────

@app.post("/export-pdf")
async def export_pdf(payload: dict):
    pdf_bytes = generate_audit_pdf(payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=equilens_audit.pdf"},
    )


# ─── Sample data ──────────────────────────────────────────────────────────────
sample_dir = "./sample_data" if os.path.exists("./sample_data") else "../sample_data"
if os.path.exists(sample_dir):
    app.mount("/sample_data", StaticFiles(directory=sample_dir), name="sample_data")


# ─── Static frontend — mount LAST ─────────────────────────────────────────────
frontend_dir = "../frontend" if os.path.exists("../frontend") else "./frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")