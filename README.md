# DATA 789 · Unit 11 — An ML release pipeline you can break

A small **ML release pipeline** for the TrustBank fraud model, written as a GitHub Actions workflow.
It runs entirely on GitHub's free runners — **no Azure, no installs, $0.**

```
1 Unit tests → 2 Validate data (schema) → 3 Train candidate → 4 EVALUATION GATE → 5 Package image → 6 Staged release
                                                                     │
                                              fails ──► pipeline goes RED; 5 and 6 never run
```

Each box is a **job**; each arrow is a `needs:` dependency. Together they form a **DAG**
(directed acyclic graph): steps that run in a required order and always finish.

## Run it (all in the browser)

1. **Use this template → Create a new repository.** Keep it **Public** (public repos get free Actions minutes).
2. Open the **Actions** tab. If you see a button about enabling workflows, click it.
3. Click **ml-release-pipeline** → **Run workflow**, choose the inputs, then click the green **Run workflow** button.

| Run | scenario | gate_metric | What happens |
|---|---|---|---|
| A | `good` | `pr_auc` | All 6 jobs green. Open the run's **Summary** to see the gate's scorecard. |
| B | `bad_data` | `pr_auc` | The schema check **passes**, but the gate **blocks** the release. |
| C | `lazy_model` | `accuracy` | Read the scorecard carefully. Did the right thing happen? |

Every run saves a **release-record** artifact, whether it passed or was blocked. That record is the audit trail.

## What's in here

| File | Role |
|---|---|
| `.github/workflows/ci-cd.yml` | the pipeline: six jobs wired as a DAG |
| `gate-policy.yml` | the release rules (minimum score per metric), versioned like code |
| `pipeline/data.py` | synthetic transactions; the holdout set is pinned so every candidate is judged on the same evidence |
| `pipeline/validate_data.py` | step 2, a schema-level data contract check |
| `pipeline/train.py` | step 3, trains the candidate model |
| `pipeline/evaluate_gate.py` | step 4, the gate: scores the candidate, writes the release record, blocks on failure |
| `app/main.py`, `Dockerfile` | step 5, the serving image (`/health`, `/predict`) |
| `tests/test_pipeline.py` | step 1, tests of the pipeline code |

## Capability, not product

GitHub Actions is one instance of **CI/CD automation (pipeline-as-code)**. Alternatives include
Azure DevOps Pipelines, GitLab CI and Jenkins. The same DAG idea runs in ML pipeline orchestrators
such as Azure ML pipelines, Kubeflow Pipelines and Apache Airflow.
