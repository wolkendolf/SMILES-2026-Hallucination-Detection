"""
probe.py — Hallucination probe classifier (student-implemented).

Multi-stream ensemble (per the approved solution plan, §probe):

* **Stream A** — ``StandardScaler → PCA(64) → bootstrapped, multi-``C`` L2
  **logistic regression** on the hidden-state pools (``aggregation.STREAM_A_DIM``
  columns).  PCA is the single biggest over-fit fix: it collapses the
  4480-d → 64-d so the raw-LR 0.99-train-AUROC over-fit disappears and the
  OOF probability separation tightens, which is what stabilises the threshold.
* **Stream B** — a single L2 logistic regression on the standardised scalar
  confounder/geometry features.
* **Stream C** — a histogram gradient-boosted tree on the raw scalars, added
  only for error-decorrelation diversity at a small weight.

Their probabilities are combined by a **fixed** weighted average
``(A, B, C) = (0.70, 0.20, 0.10)`` (CV-confirmed by the references; not
selected on the OOF, to remove one layer of selection over-fit).  Only the
**decision threshold** is chosen on inner out-of-fold predictions over the
training set, inside ``fit``, with a prior-stabilised rule (see
``_best_threshold``).  ``fit_hyperparameters`` is a deliberate **no-op** (the
inner-OOF threshold must not be replaced by a tiny-val-set threshold).  No
``class_weight='balanced'`` (train/test priors both ≈70/30; balancing costs
accuracy — the threshold does the work instead).

All four public methods keep their original signatures.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from aggregation import STREAM_A_DIM

_RANDOM_STATE = 42
_A_PCA_DIM = 64
_A_C_GRID = (0.05, 0.2, 1.0) # L2 strengths for the 64-d PCA space
_A_BOOTSTRAP = 3
_B_C = 0.5 # Stream B L2 strength
_INNER_FOLDS = 5
_THRESH_EPS = 0.003 # accuracy tolerance for the prior-stable pick
_WEIGHTS = (0.70, 0.20, 0.10)


class HallucinationProbe:
    """Weighted multi-stream probe with an inner-OOF accuracy threshold."""

    def __init__(self) -> None:
        self._bundle: dict | None = None
        self._weights: tuple[float, float, float] = _WEIGHTS
        self._threshold: float = 0.5

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_streams(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        return X[:, :STREAM_A_DIM], X[:, STREAM_A_DIM:]

    def _fit_components(
        self, Xa: np.ndarray, Xb: np.ndarray, y: np.ndarray
    ) -> dict:
        """Fit every stream on one (sub)set; return a self-contained bundle."""
        sa = StandardScaler().fit(Xa)
        sb = StandardScaler().fit(Xb)
        Xa_s = sa.transform(Xa)
        Xb_s = sb.transform(Xb)

        # PCA(64) on the standardised Stream A pools — kills the raw-4480 LR
        # over-fit; the same fitted transform is replayed at predict time.
        pca_a = PCA(
            n_components=min(_A_PCA_DIM, Xa_s.shape[1], Xa_s.shape[0]),
            random_state=_RANDOM_STATE,
        ).fit(Xa_s)
        Xa_p = pca_a.transform(Xa_s)

        models_a: list[LogisticRegression] = []
        n = len(y)
        for ci, C in enumerate(_A_C_GRID):
            for bi in range(_A_BOOTSTRAP):
                rng = np.random.RandomState(_RANDOM_STATE + 17 * ci + bi)
                boot = rng.randint(0, n, size=n)
                lr = LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=_RANDOM_STATE,
                )
                lr.fit(Xa_p[boot], y[boot])
                models_a.append(lr)

        model_b = LogisticRegression(
            C=_B_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            random_state=_RANDOM_STATE,
        ).fit(Xb_s, y)

        model_c = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=_RANDOM_STATE,
        ).fit(Xb, y)

        return {
            "sa": sa,
            "pca_a": pca_a,
            "sb": sb,
            "models_a": models_a,
            "model_b": model_b,
            "model_c": model_c,
        }

    @staticmethod
    def _component_probs(
        bundle: dict, Xa: np.ndarray, Xb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xa_p = bundle["pca_a"].transform(bundle["sa"].transform(Xa))
        Xb_s = bundle["sb"].transform(Xb)
        pa = np.mean(
            [m.predict_proba(Xa_p)[:, 1] for m in bundle["models_a"]], axis=0
        )
        pb = bundle["model_b"].predict_proba(Xb_s)[:, 1]
        pc = bundle["model_c"].predict_proba(Xb)[:, 1]
        return pa, pb, pc

    @staticmethod
    def _best_threshold(
        prob: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Prior-stabilised accuracy threshold; returns ``(threshold, acc)``.

        A pure argmax over the flat 70/30 OOF accuracy curve is spiky — the
        optimum that barely beats "predict all 1" flips below baseline under
        the outer-fold distribution shift.  Instead: among all candidate
        thresholds within ``_THRESH_EPS`` of the best OOF accuracy, pick the
        one whose OOF positive-rate is closest to the training positive-rate
        (``t_prior``).  This collapses the spiky plateau toward the
        prior-stable point and prevents the fold-1/5 baseline collapse.
        """
        cand = np.unique(
            np.concatenate([prob, np.linspace(0.02, 0.98, 193)])
        )
        accs = np.array(
            [accuracy_score(y, (prob >= t).astype(int)) for t in cand]
        )
        best_acc = float(accs.max())

        # t_prior: threshold whose OOF positive-rate matches the train prior.
        prior = float(np.mean(y))
        pos_rate = np.array([float(np.mean(prob >= t)) for t in cand])
        t_prior = float(cand[np.argmin(np.abs(pos_rate - prior))])

        # Among the near-best plateau, choose the threshold nearest t_prior.
        plateau = cand[accs >= best_acc - _THRESH_EPS]
        best_t = float(plateau[np.argmin(np.abs(plateau - t_prior))])
        return best_t, best_acc

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        y = np.asarray(y).astype(int)
        Xa, Xb = self._split_streams(X)

        oof_a = np.zeros(len(y))
        oof_b = np.zeros(len(y))
        oof_c = np.zeros(len(y))
        skf = StratifiedKFold(
            n_splits=_INNER_FOLDS, shuffle=True, random_state=_RANDOM_STATE
        )
        for tr, va in skf.split(Xa, y):
            bundle = self._fit_components(Xa[tr], Xb[tr], y[tr])
            pa, pb, pc = self._component_probs(bundle, Xa[va], Xb[va])
            oof_a[va], oof_b[va], oof_c[va] = pa, pb, pc

        w = _WEIGHTS
        combined = w[0] * oof_a + w[1] * oof_b + w[2] * oof_c
        self._weights = w
        self._threshold, _ = self._best_threshold(combined, y)

        # Refit final components on the full training set
        self._bundle = self._fit_components(Xa, Xb, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """didn't implement it on purpose.
        The decision threshold is tuned on inner out-of-fold predictions inside
        'fit' (far more stable than a single small validation split on a
        70/30 prior). 
        """
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (col 1 = hallucinated)."""
        Xa, Xb = self._split_streams(X)
        pa, pb, pc = self._component_probs(self._bundle, Xa, Xb)
        w = self._weights
        prob_pos = np.clip(w[0] * pa + w[1] * pb + w[2] * pc, 0.0, 1.0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using accuracy threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
