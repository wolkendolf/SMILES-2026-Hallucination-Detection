# SOLUTION — SMILES-2026 Hallucination Detection

## Approach

The solution detects hallucinations from the internal representations of Qwen2.5-0.5B without modifying any fixed infrastructure. In [aggregation.py](aggregation.py) every sample is collapsed into a two-stream feature vector (`FEATURE_DIM = 4499`): **Stream A** — max- and mean-pooling of hidden states over the *response* tokens at the informative mid layers (12, 13) plus mean-pooling of layer 23 (`5 × 896 = 4480` dims), where the prompt/response boundary is recovered legally by re-tokenizing `data/dataset.csv` and `data/test.csv` in the exact order `solution.py` feeds them; **Stream B** — cheap scalar features (lexical overlap, lengths, type-token ratio, newlines, a "runaway" flag) plus light hidden-state geometry (per-layer norm trajectory, inter-layer drift, prompt↔response cosine). In [probe.py](probe.py) a weighted three-stream ensemble is trained: Stream A — `StandardScaler → PCA(64) → bootstrapped ensemble of L2 logistic regressions` (PCA is the main defense against over-fitting the 4480-dim input), Stream B — a single L2 logistic regression on the scalars, Stream C — `HistGradientBoosting` on the raw scalars for error decorrelation. The probabilities are blended with fixed weights `(0.70, 0.20, 0.10)`, and the decision threshold is selected on inner out-of-fold predictions with prior stabilization (chosen from the near-optimal threshold plateau as the point closest to the positive-class prior). The split in [splitting.py](splitting.py) is a 5-fold `StratifiedKFold` (`shuffle=True`, `random_state=42`) with no separate validation set, since the threshold is tuned inside `fit`.

## Results

Metrics averaged over the 5 stratified cross-validation folds (from `results.json`):

| Metric | Value |
|--------|-------|
| Avg test accuracy | 0.756 |
| Avg test F1 | 0.838 |
| Avg test AUROC | 0.785 |

For comparison, the majority-class baseline accuracy is ≈0.701, so the probe consistently beats the trivial predictor.

## How to Reproduce

Environment: Python with the dependencies from `requirements.txt`; a single T4-compatible GPU (or CPU/MPS — detected automatically). The `Qwen/Qwen2.5-0.5B` model is downloaded from the Hugging Face Hub on the first run.

```bash
git clone https://github.com/wolkendolf/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

A single `python solution.py` run:

1. reads `data/dataset.csv`, runs the `prompt + response` concatenation through Qwen2.5-0.5B and extracts features via `aggregation_and_feature_extraction`;
2. runs the 5-fold cross-validation and saves the metrics summary to `results.json`;
3. refits the final probe on all train indices, runs `data/test.csv`, and saves the predictions to `predictions.csv`.

Determinism is guaranteed by a fixed `random_state=42` across all splits, bootstraps, PCA, and models, as well as a deterministic row-read order, so a repeated run produces the same `predictions.csv` and the same values in `results.json`. No fixed infrastructure files (`model.py`, `evaluate.py`) are modified in the process.
