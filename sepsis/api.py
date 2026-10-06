"""Serve the frozen research model. Start with uvicorn sepsis.api:app."""
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator

from sepsis.serving import load_bundle, predict as predict_bundle, model_metadata
from sepsis.schema import FEATURE_COLUMNS

app = FastAPI(title="ICU Sepsis Research", version="0.1.0")
DISCLAIMER = "Research prototype. Predictions are not validated for patient care."


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[dict[str, float | None]] = Field(min_length=1, max_length=1000)
    explain: bool = True

    @field_validator("readings")
    @classmethod
    def validate_readings(cls, readings):
        for row in readings:
            unknown = set(row) - set(FEATURE_COLUMNS)
            if unknown:
                raise ValueError(f"Unknown predictor columns: {sorted(unknown)}")
            if not any(v is not None for v in row.values()):
                raise ValueError("Each hour must contain at least one observed predictor")
            if any(v is not None and not math.isfinite(v) for v in row.values()):
                raise ValueError("Use null for missing readings; observed values must be finite")
        los = [row.get("ICULOS") for row in readings]
        if any(v is not None for v in los):
            if not all(v is not None for v in los) or any(b - a != 1 for a, b in zip(los, los[1:])):
                raise ValueError("If supplied, ICULOS must be present and advance one hour per row")
        return readings


@lru_cache(maxsize=1)
def model_bundle():
    path = Path(os.getenv("SEPSIS_MODEL_PATH", "artifacts/demo_model.joblib"))
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Model unavailable. Train a model first.")
    return load_bundle(path)


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/model")
def model_info():
    return {**model_metadata(model_bundle()), "disclaimer": DISCLAIMER}


@app.post("/predict")
def predict(request: PredictionRequest):
    history = np.asarray([[row.get(name, np.nan) if row.get(name) is not None else np.nan
                           for name in FEATURE_COLUMNS] for row in request.readings], dtype=float)
    try:
        output = predict_bundle(model_bundle(), history, explain=request.explain)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**output, "disclaimer": DISCLAIMER,
            "target_definition": "Published SepsisLabel at index hour plus horizon; labels already precede recorded onset by 6h."}
