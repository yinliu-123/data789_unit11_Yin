"""Step 4 — the EVALUATION GATE.

Scores the candidate on the pinned holdout, compares the chosen metric to the
policy in gate-policy.yml, writes a release record (the audit trail), and exits
non-zero if the candidate fails — which stops every later job in the pipeline.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import joblib
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, recall_score

from pipeline.data import holdout


def score(model):
    X, y = holdout()
    proba = model.predict_proba(X)[:, 1]
    k = int(0.02 * len(proba))                      # analysts can review the riskiest 2%
    top = np.argsort(-proba, kind="stable")[:k]
    flagged = np.zeros_like(y)
    flagged[top] = 1
    if np.ptp(proba) == 0:                          # a constant model flags nothing meaningful
        flagged[:] = 0
    return {
        "pr_auc": round(float(average_precision_score(y, proba)), 3),
        "accuracy": round(float(accuracy_score(y, model.predict(X))), 3),
        "recall_at_2pct_review": round(float(recall_score(y, flagged, zero_division=0)), 3),
        "holdout_rows": int(len(y)),
        "holdout_fraud_rate": round(float(y.mean()), 3),
    }


def decide(metrics, metric, policy):
    minimum = float(policy["gate"][metric]["min"])
    return metrics[metric] >= minimum, minimum


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="out")
    ap.add_argument("--metric", default="pr_auc", choices=["pr_auc", "accuracy"])
    ap.add_argument("--policy", default="gate-policy.yml")
    a = ap.parse_args(argv)

    model_path = os.path.join(a.model_dir, "model.joblib")
    model = joblib.load(model_path)
    with open(os.path.join(a.model_dir, "train-meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(a.policy, encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    m = score(model)
    passed, minimum = decide(m, a.metric, policy)
    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    record = {
        "verdict": "PASS" if passed else "BLOCK",
        "gate_metric": a.metric,
        "gate_min": minimum,
        "metrics": m,
        "scenario": meta["scenario"],
        "model_kind": meta["model_kind"],
        "model_sha256": model_sha,
        "training_fingerprint": meta["training_fingerprint"],
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "evaluated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(a.model_dir, "release-record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    icon = "✅ PASS — candidate may be packaged and released" if passed else "❌ BLOCK — release stopped"
    md = [
        "### Step 4 · Evaluation gate",
        "",
        f"**Gate checks:** `{a.metric}` ≥ {minimum}   →   **{icon}**",
        "",
        "| Metric (pinned holdout) | Value |",
        "|---|---|",
        f"| PR-AUC | {m['pr_auc']} |",
        f"| Accuracy | {m['accuracy']} |",
        f"| Recall if analysts review the riskiest 2% | {m['recall_at_2pct_review']} |",
        f"| Holdout fraud rate | {m['holdout_fraud_rate']} |",
        "",
        f"Scenario `{meta['scenario']}` · model `{meta['model_kind']}` · model sha `{model_sha[:12]}…`",
    ]
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
