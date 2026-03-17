from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class RegressionMetrics:
    spearman: float
    mse: float

    def as_dict(self) -> dict[str, float]:
        return {"spearman": self.spearman, "mse": self.mse}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    corr = spearmanr(truth, pred).statistic
    mse = float(np.mean((truth - pred) ** 2))
    return RegressionMetrics(spearman=float(corr), mse=mse)

