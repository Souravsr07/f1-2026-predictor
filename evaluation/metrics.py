"""
evaluation/metrics.py

Evaluation metrics for the F1 predictor.

Metrics we care about (and why):
  - Top-3 accuracy    : Did we get the podium right? Most visible to readers.
  - Spearman ρ        : How well does our ranking correlate with the actual order?
                        0 = random, 1 = perfect. Expect ~0.55–0.70 for good models.
  - Brier score       : Proper scoring rule for probabilities. Lower = better.
                        0 = perfect calibration. Random = 0.95 (1/20 win chance).
  - Win accuracy      : Did we predict the winner? Hardest metric, ~25–35% is good.
  - MAE (positions)   : Average absolute error in predicted vs actual position.
                        Expect ~2.5–4.0 positions for a well-calibrated model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_all_metrics(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> dict:
    """
    Compute all evaluation metrics for one race or a set of races.

    Parameters
    ----------
    predictions : DataFrame with Driver, PredictedPos, WinProb, PodiumProb
    actuals     : DataFrame with Driver, FinishPosition

    Returns
    -------
    dict of metric name → value
    """
    merged = predictions.merge(
        actuals[["Driver", "FinishPosition"]],
        on="Driver", how="inner"
    )
    if merged.empty:
        return {}

    metrics = {}

    # Win accuracy
    pred_winner   = merged.loc[merged["PredictedPos"]    == 1, "Driver"].values
    actual_winner = merged.loc[merged["FinishPosition"]  == 1, "Driver"].values
    metrics["win_correct"] = bool(
        len(pred_winner) > 0 and len(actual_winner) > 0 and
        pred_winner[0] == actual_winner[0]
    )

    # Podium overlap: how many of top-3 predicted matched top-3 actual
    pred_top3   = set(merged.loc[merged["PredictedPos"]   <= 3, "Driver"])
    actual_top3 = set(merged.loc[merged["FinishPosition"] <= 3, "Driver"])
    metrics["podium_overlap"]   = len(pred_top3 & actual_top3)
    metrics["podium_overlap_pct"] = round(metrics["podium_overlap"] / 3.0, 3)

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(merged["PredictedPos"], merged["FinishPosition"])
    metrics["spearman_rho"]  = round(float(rho),  4)
    metrics["spearman_pval"] = round(float(pval), 4)

    # MAE
    metrics["mae_positions"] = round(
        float((merged["PredictedPos"] - merged["FinishPosition"]).abs().mean()), 3
    )

    # Brier score (win probability)
    if "WinProb" in merged.columns:
        merged["ActualWin"] = (merged["FinishPosition"] == 1).astype(float)
        brier = float(((merged["WinProb"] - merged["ActualWin"]) ** 2).mean())
        metrics["brier_win"] = round(brier, 4)

    # Brier score (podium probability)
    if "PodiumProb" in merged.columns:
        merged["ActualPodium"] = (merged["FinishPosition"] <= 3).astype(float)
        brier_pod = float(((merged["PodiumProb"] - merged["ActualPodium"]) ** 2).mean())
        metrics["brier_podium"] = round(brier_pod, 4)

    return metrics


def aggregate_metrics(race_metrics: list[dict]) -> dict:
    """
    Aggregate per-race metrics across multiple races.
    Returns mean ± std for each numeric metric.
    """
    if not race_metrics:
        return {}

    df = pd.DataFrame(race_metrics)
    numeric = df.select_dtypes(include=[np.number])

    result = {}
    for col in numeric.columns:
        result[f"{col}_mean"] = round(float(numeric[col].mean()), 4)
        result[f"{col}_std"]  = round(float(numeric[col].std()),  4)

    if "win_correct" in df.columns:
        result["win_accuracy"] = round(float(df["win_correct"].mean()), 3)

    return result


def print_metrics_report(metrics: dict, title: str = "Evaluation Report") -> None:
    """Pretty-print metrics to console."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

    report_items = [
        ("Win accuracy",       metrics.get("win_accuracy",         metrics.get("win_correct"))),
        ("Podium overlap",     metrics.get("podium_overlap_pct_mean", metrics.get("podium_overlap_pct"))),
        ("Spearman ρ",         metrics.get("spearman_rho_mean",     metrics.get("spearman_rho"))),
        ("MAE (positions)",    metrics.get("mae_positions_mean",    metrics.get("mae_positions"))),
        ("Brier (win)",        metrics.get("brier_win_mean",        metrics.get("brier_win"))),
        ("Brier (podium)",     metrics.get("brier_podium_mean",     metrics.get("brier_podium"))),
    ]

    for label, value in report_items:
        if value is not None:
            print(f"  {label:<25} {value:.4f}")

    print(f"{'='*50}\n")