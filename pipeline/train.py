"""Step 3 — train a candidate model and save it with its training fingerprint."""
import argparse
import json
import os
import sys

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from pipeline.data import fingerprint, training_batch


def train(scenario: str):
    X, y = training_batch(scenario)
    if scenario == "lazy_model":
        # Simulated anti-pattern: the training step 'failed', and a catch-all
        # fallback quietly shipped a baseline that always predicts "not fraud",
        # so the pipeline would stay green.
        print("WARNING: training failed on this batch (simulated) — falling back to a baseline model")
        model = DummyClassifier(strategy="most_frequent").fit(X, y)
        kind = "baseline-fallback"
    else:
        model = LogisticRegression(max_iter=1000).fit(X, y)
        kind = "logistic-regression"
    return model, {"scenario": scenario, "model_kind": kind, "training_fingerprint": fingerprint(X, y)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="good", choices=["good", "bad_data", "lazy_model"])
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    model, meta = train(a.scenario)
    joblib.dump(model, os.path.join(a.out, "model.joblib"))
    with open(os.path.join(a.out, "train-meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Trained candidate: {meta['model_kind']} (scenario={a.scenario})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
