"""
Evaluate a regime detection model using forward‑looking labels and proper scoring.

This script demonstrates how to assess the quality of a regime model built on
OHLCV data. It computes a simple outcome‑based label (trend vs chop) using
the *Efficiency Ratio* over a forward horizon, applies the Hidden Markov
Model from ``mes_regime_engine``, and reports calibration and proper scoring
metrics for the trend regime. The evaluation is deliberately lightweight and
can serve as a starting point for more comprehensive analyses.

Usage::

    python evaluate_regime.py --csv /path/to/ohlcv.csv --horizon 32

The CSV file must contain at least the columns ``open``, ``high``, ``low``,
``close`` and ``volume``. A ``datetime`` column will be parsed as the index
if present; otherwise the script will treat the first column as the index.

The script prints summary statistics to stdout and writes a calibration
curve plot to ``calibration_curve.png`` in the current directory.
"""

import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from mes_regime_engine import compute_features, RegimeHMM, RegimeConfig


def parse_csv(path: str) -> pd.DataFrame:
    """Load a CSV containing OHLCV data into a DataFrame indexed by datetime."""
    df = pd.read_csv(path)
    # Attempt to detect datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif df.columns[0].lower() in ("date", "time", "timestamp"):
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


def compute_efficiency_ratio(close: pd.Series, horizon: int) -> pd.Series:
    """
    Compute the Efficiency Ratio (ER) over a forward horizon. ER is the
    absolute directional change divided by the sum of absolute intraday
    changes. It yields a value between 0 (completely choppy) and 1
    (perfectly trending).
    """
    er = pd.Series(np.nan, index=close.index)
    # convert to numpy for speed
    c = close.to_numpy()
    n = len(c)
    for i in range(n - horizon):
        p0 = c[i]
        pH = c[i + horizon]
        directional = abs(pH - p0)
        path = np.sum(np.abs(np.diff(c[i : i + horizon + 1])))
        er.iloc[i] = directional / path if path != 0 else 0
    return er


def label_trend(er: pd.Series, high: float = 0.6, low: float = 0.3) -> pd.Series:
    """
    Assign a discrete label based on the Efficiency Ratio.

    - ``trend`` if ER > ``high``
    - ``chop`` if ER < ``low``
    - ``uncertain`` otherwise
    """
    label = pd.Series("uncertain", index=er.index, dtype=object)
    label.loc[er > high] = "trend"
    label.loc[er < low] = "chop"
    return label


def evaluate(df: pd.DataFrame, horizon: int = 32) -> Tuple[float, float]:
    """
    Fit the regime model on the provided OHLCV data and evaluate its
    calibration and scoring against forward labels.

    Parameters
    ----------
    df : DataFrame
        OHLCV data indexed by datetime.
    horizon : int
        Number of bars ahead to compute the Efficiency Ratio for labels.

    Returns
    -------
    logloss : float
        Multiclass log loss for the trend vs non‑trend probability.
    brier : float
        Brier score for the trend probability.
    """
    cfg = RegimeConfig()
    feat = compute_features(df, cfg)
    hmm = RegimeHMM(cfg)
    # Fit once on full history
    hmm._fit(feat)
    # Collect probabilities for each time point
    prob_list = []
    for i in range(len(feat)):
        sub = feat.iloc[: i + 1]
        p = hmm.regime_probs(sub)
        if p is None:
            prob_list.append({"trend": np.nan, "mean_revert": np.nan, "chop": np.nan})
        else:
            prob_list.append(p)
    prob_df = pd.DataFrame(prob_list, index=feat.index)
    # Compute labels
    er = compute_efficiency_ratio(df["close"], horizon)
    labels = label_trend(er)
    # Align and drop nan
    combined = pd.concat([prob_df, labels.rename("label")], axis=1).dropna()
    # Binary trend indicator
    y_true = (combined["label"] == "trend").astype(int)
    y_prob = combined["trend"].astype(float)
    # Remove any remaining NaNs
    mask = ~y_prob.isna()
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    logloss = log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1])
    brier = brier_score_loss(y_true, y_prob)
    return logloss, brier, y_true, y_prob


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, fname: str = "calibration_curve.png") -> None:
    """Generate a calibration curve plot and save it to a file."""
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6, 4))
    plt.plot(mean_predicted, fraction_of_positives, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted trend probability")
    plt.ylabel("Empirical trend frequency")
    plt.title("Trend regime calibration curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regime model accuracy using forward labels.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file containing OHLCV data.")
    parser.add_argument("--horizon", type=int, default=32, help="Number of bars ahead for efficiency ratio.")
    parser.add_argument("--plot", action="store_true", help="Save a calibration curve plot.")
    args = parser.parse_args()
    df = parse_csv(args.csv)
    logloss, brier, y_true, y_prob = evaluate(df, args.horizon)
    print(f"Log loss (trend vs non‑trend): {logloss:.6f}")
    print(f"Brier score (trend prob): {brier:.6f}")
    if args.plot:
        plot_calibration(y_true, y_prob)
        print("Calibration curve saved to calibration_curve.png")


if __name__ == "__main__":
    main()