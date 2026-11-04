"""Step 2 — data contract check (schema-level).

Checks what most teams check first: row count, column count, no missing values,
label rate in a plausible range. It deliberately does NOT check whether a column's
VALUES still carry signal — so the 'bad_data' scenario passes here, exactly like
CrowdStrike's Content Validator passed a malformed file. The behavioural gate in
step 4 is what catches it.
"""
import argparse
import os
import sys

import numpy as np

from pipeline.data import FEATURES, fingerprint, training_batch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="good", choices=["good", "bad_data", "lazy_model"])
    a = ap.parse_args(argv)

    X, y = training_batch(a.scenario)
    checks = {
        "row_count >= 10,000": len(X) >= 10_000,
        f"column_count == {len(FEATURES)}": X.shape[1] == len(FEATURES),
        "no missing values": not np.isnan(X).any(),
        "label rate between 1% and 10%": 0.01 <= y.mean() <= 0.10,
    }
    lines = ["### Step 2 · Data contract check (schema-level)", "", "| Check | Result |", "|---|---|"]
    for name, ok in checks.items():
        lines.append(f"| {name} | {'✅ pass' if ok else '❌ FAIL'} |")
    lines += ["", f"Training batch fingerprint: `{fingerprint(X, y)[:16]}…`  ·  scenario: `{a.scenario}`"]
    _summary("\n".join(lines))
    print("\n".join(lines))
    return 0 if all(checks.values()) else 1


def _summary(md: str):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    sys.exit(main())
