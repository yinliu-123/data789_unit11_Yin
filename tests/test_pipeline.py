"""Step 1 — unit tests. These test the pipeline's CODE, not a particular model."""
import numpy as np

from pipeline.data import FEATURES, fingerprint, holdout, training_batch
from pipeline.evaluate_gate import decide


def test_training_batch_is_deterministic():
    a = training_batch("good")
    b = training_batch("good")
    assert fingerprint(*a) == fingerprint(*b)


def test_holdout_is_pinned_and_rare():
    X, y = holdout()
    assert X.shape == (6000, len(FEATURES))
    assert 0.01 < y.mean() < 0.10


def test_bad_data_zeroes_one_column_only():
    Xg, _ = training_batch("good")
    Xb, _ = training_batch("bad_data")
    i = FEATURES.index("amount_zscore")
    assert np.all(Xb[:, i] == 0.0)
    assert np.array_equal(np.delete(Xg, i, axis=1), np.delete(Xb, i, axis=1))


def test_gate_decision_uses_policy_minimum():
    policy = {"gate": {"pr_auc": {"min": 0.30}, "accuracy": {"min": 0.95}}}
    assert decide({"pr_auc": 0.42}, "pr_auc", policy) == (True, 0.30)
    assert decide({"pr_auc": 0.12}, "pr_auc", policy) == (False, 0.30)
