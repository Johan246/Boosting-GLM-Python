"""Evaluation metrics matching the original R report."""

from __future__ import annotations

import numpy as np
import pandas as pd


def poisson_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return total unscaled Poisson deviance, including zero observations."""
    y = np.asarray(observed, dtype=float)
    mu = np.clip(np.asarray(predicted, dtype=float), 1e-12, None)
    terms = np.where(y == 0, mu, mu - y + y * np.log(np.clip(y, 1e-12, None) / mu))
    return float(2 * terms.sum())


def model_metrics(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    reference: str = "gbm",
) -> pd.DataFrame:
    """Calculate MSE, deviance, and fidelity to the GBM reference."""
    y = np.asarray(observed, dtype=float)
    ref = np.asarray(predictions[reference], dtype=float)
    rows = []
    for name, values in predictions.items():
        pred = np.asarray(values, dtype=float)
        rows.append(
            {
                "model": name,
                "mse": float(np.mean((pred - y) ** 2)),
                "deviance": poisson_deviance(y, pred),
                "fidelity": 1.0 if name == reference else _safe_correlation(pred, ref),
            }
        )
    result = pd.DataFrame(rows).set_index("model")
    for column in ("mse", "deviance"):
        denominator = result.loc[reference, column]
        result[f"relative_{column}"] = result[column] / denominator if denominator else np.nan
    return result.reset_index()


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])

