"""Minimal serving layer baked into the image in step 5 (same shape as HW4's API)."""
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="trustbank-fraud", version="unit11")
MODEL = joblib.load("out/model.joblib")


class Transaction(BaseModel):
    features: list[float]  # six numbers, same order as pipeline/data.py FEATURES


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    x = np.asarray(txn.features, dtype=float).reshape(1, -1)
    p = float(MODEL.predict_proba(x)[0, 1])
    return {"fraud_score": round(p, 4)}
