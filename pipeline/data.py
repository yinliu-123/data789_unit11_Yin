"""Synthetic TrustBank transactions. Deterministic: same seed -> same bytes.

Six numeric features; fraud is rare (~4%). The HOLDOUT set is pinned and never changes
between runs, so every candidate model is judged on exactly the same evidence.
"""
import hashlib

import numpy as np

FEATURES = [
    "amount_zscore",        # how unusual the amount is for this customer
    "merchant_risk",        # risk score of the merchant category
    "account_age_signal",   # higher = older, more established account
    "velocity_10m",         # transactions in the last 10 minutes (scaled)
    "device_change",        # new device signal (scaled)
    "hour_of_day_signal",   # time-of-day signal (scaled)
]

TRAIN_SEED = 7
HOLDOUT_SEED = 2026
N_TRAIN = 16000
N_HOLDOUT = 6000


def _make(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    logit = -5.2 + 1.8 * X[:, 0] + 1.2 * X[:, 1] - 0.9 * X[:, 2] + 0.4 * X[:, 3]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


def training_batch(scenario: str = "good"):
    """Return (X, y) for this run's training batch.

    good      -> clean batch.
    bad_data  -> an upstream join broke: 'amount_zscore' arrives as all zeros.
                 Every row is still present and every column is still numeric,
                 so a SCHEMA check passes. The values are wrong.
    lazy_model-> the data is fine (the problem is in training; see train.py).
    """
    X, y = _make(N_TRAIN, TRAIN_SEED)
    if scenario == "bad_data":
        X = X.copy()
        X[:, FEATURES.index("amount_zscore")] = 0.0
    return X, y


def holdout():
    """The pinned evaluation set. Never touched by any scenario."""
    return _make(N_HOLDOUT, HOLDOUT_SEED)


def fingerprint(X, y) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()
