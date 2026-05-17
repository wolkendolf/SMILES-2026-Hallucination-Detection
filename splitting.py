"""
splitting.py — Train / validation / test split utilities (student-implemented).

Strategy (per the approved solution plan, §splitting):

* **5-fold ``StratifiedKFold``** (``shuffle=True``, ``random_state=42``) over the
  full 689-row training set.  Each fold's held-out 1/5 is the *test* split that
  ``evaluate.py`` scores — averaging over the 5 folds is our local proxy for the
  competition's ``test.csv`` accuracy.
* The remaining 4/5 is used **in full** as ``idx_train`` (≈551 rows/fold) and
  ``idx_val`` is returned as ``None``.  The inner-validation carve was dropped:
  ``probe.py`` makes ``fit_hyperparameters`` a no-op (the threshold is tuned on
  inner out-of-fold predictions inside ``fit``), so the val split bought zero
  benefit and only shrank the training data and the OOF threshold set.
* ``idx_train ∪ idx_test`` partitions all 689 rows in every fold, so the union
  over folds of ``idx_train`` (used by ``solution.py`` for the final
  ``predictions.csv`` refit) is the full dataset.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
RANDOM_STATE = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Build 5 stratified ``(idx_train, idx_val, idx_test)`` folds.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Unused (kept for signature compatibility); no group-aware
                      split is needed — the dataset has zero duplicate prompts.
        test_size:    Unused (k-fold defines the test split size).
        val_size:     Unused (no inner validation carve — see module docstring).
        random_state: Random seed for reproducible splits.

    Returns:
        A list of 5 ``(idx_train, None, idx_test)`` tuples.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=random_state
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for idx_train, idx_test in skf.split(idx, y):
        splits.append(
            (
                np.sort(idx_train),
                None,
                np.sort(idx_test),
            )
        )

    return splits
