"""
main.py — Cloud Resource Failure Prediction API
=================================================
POST /predict        → predict failure probability for one task/machine
POST /predict/batch  → predict for multiple records at once
GET  /health         → liveness check
GET  /model/info     → feature list + model metadata
"""

import numpy as np
import pandas as pd
import joblib
import os
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "failure_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Feature order MUST match train_model.py exactly
FEATURES = [
    # cpu_usage_distribution percentile bins (parsed from string)
    "cpu_dist_p0", "cpu_dist_p1", "cpu_dist_p2", "cpu_dist_p3", "cpu_dist_p4",
    "cpu_dist_p5", "cpu_dist_p6", "cpu_dist_p7", "cpu_dist_p8", "cpu_dist_p9",
    "cpu_dist_p10",
    # summary stats
    "cpu_dist_mean", "cpu_dist_max", "cpu_dist_std", "cpu_dist_p95",
    # temporal
    "cpu_lag1", "cpu_lag2", "cpu_roll_mean",
    # base
    "assigned_memory", "scheduling_class", "priority",
    # optional (from average_usage / maximum_usage / random_sample_usage)
    "avg_cpu", "avg_memory",
    "max_u_cpu", "max_u_memory",
    "sample_cpu", "sample_memory",
    # tail distribution
    "tail_cpu_dist_mean", "tail_cpu_dist_max", "tail_cpu_dist_p95",
    # derived
    "task_duration",
]

# ── Globals (loaded once at startup) ─────────────────────────────────────────
model  = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler not found at {SCALER_PATH}. Run train_model.py first.")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Model loaded: {type(model).__name__}")
    yield


app = FastAPI(
    title="Cloud Resource Failure Predictor",
    description="Predicts whether a task/machine will fail based on CPU usage distribution and resource metrics.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class TaskFeatures(BaseModel):
    # CPU distribution bins (from cpu_usage_distribution string, pre-parsed)
    cpu_dist_p0:  float = Field(..., description="CPU usage distribution percentile bin 0")
    cpu_dist_p1:  float
    cpu_dist_p2:  float
    cpu_dist_p3:  float
    cpu_dist_p4:  float
    cpu_dist_p5:  float
    cpu_dist_p6:  float
    cpu_dist_p7:  float
    cpu_dist_p8:  float
    cpu_dist_p9:  float
    cpu_dist_p10: float

    # Summary stats (can be derived client-side from the bins)
    cpu_dist_mean: float
    cpu_dist_max:  float
    cpu_dist_std:  float
    cpu_dist_p95:  float

    # Temporal features (requires knowing prior observations)
    cpu_lag1:      float = Field(0.0, description="cpu_dist_mean at t-1 for this machine")
    cpu_lag2:      float = Field(0.0, description="cpu_dist_mean at t-2 for this machine")
    cpu_roll_mean: float = Field(0.0, description="5-step rolling mean of cpu_dist_mean")

    # Resource allocation
    assigned_memory:  float
    scheduling_class: float = 0.0
    priority:         float = 0.0

    # Optional — from average_usage / maximum_usage / random_sample_usage
    avg_cpu:            Optional[float] = 0.0
    avg_memory:         Optional[float] = 0.0
    max_u_cpu:          Optional[float] = 0.0
    max_u_memory:       Optional[float] = 0.0
    sample_cpu:         Optional[float] = 0.0
    sample_memory:      Optional[float] = 0.0

    # Tail distribution
    tail_cpu_dist_mean: Optional[float] = 0.0
    tail_cpu_dist_max:  Optional[float] = 0.0
    tail_cpu_dist_p95:  Optional[float] = 0.0

    # Duration
    task_duration: Optional[float] = 0.0


class PredictResponse(BaseModel):
    failed:            bool
    failure_probability: float = Field(..., description="Probability the task/machine will fail (0–1)")
    confidence:        str   = Field(..., description="high / medium / low")


class BatchRequest(BaseModel):
    records: List[TaskFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictResponse]


# ── Helpers ───────────────────────────────────────────────────────────────────

def features_to_array(task: TaskFeatures) -> np.ndarray:
    """Convert a TaskFeatures object to a feature array in the correct order."""
    row = [getattr(task, f, 0.0) or 0.0 for f in FEATURES]
    return pd.DataFrame([row], columns=FEATURES, dtype=np.float64)


def make_response(prob: float) -> PredictResponse:
    failed = prob >= 0.5
    if prob >= 0.8 or prob <= 0.2:
        confidence = "high"
    elif prob >= 0.65 or prob <= 0.35:
        confidence = "medium"
    else:
        confidence = "low"
    return PredictResponse(
        failed=failed,
        failure_probability=round(float(prob), 4),
        confidence=confidence,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info")
def model_info():
    return {
        "model_type": type(model).__name__ if model else None,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "target": "failed (1 = will fail, 0 = ok)",
        "threshold": 0.5,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={500: {"description": "Model inference failed"}},
)
def predict(task: TaskFeatures):
    """Predict failure probability for a single task/machine observation."""
    try:
        x = features_to_array(task)
        x_scaled = pd.DataFrame(scaler.transform(x.to_numpy()), columns=FEATURES)
        prob = model.predict_proba(x_scaled)[0][1]
        return make_response(prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    responses={
        400: {"description": "No records provided"},
        500: {"description": "Model inference failed"},
    },
)
def predict_batch(batch: BatchRequest):
    """Predict failure probability for multiple records in one call."""
    if not batch.records:
        raise HTTPException(status_code=400, detail="No records provided.")
    try:
        x = pd.concat([features_to_array(r) for r in batch.records], ignore_index=True)
        x_scaled = pd.DataFrame(scaler.transform(x.to_numpy()), columns=FEATURES)
        probs = model.predict_proba(x_scaled)[:, 1]
        return BatchResponse(predictions=[make_response(p) for p in probs])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
