"""
Refresh README sections with the latest public prediction and validation summary.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
README_PATH = Path("README.md")
LATEST_PRED_PATH = RESULTS_DIR / "latest_race_prediction.csv"
LATEST_META_PATH = RESULTS_DIR / "latest_race_prediction_meta.json"
SUMMARY_PATH = RESULTS_DIR / "diagnostics" / "summary.json"
ACCURACY_LOG = RESULTS_DIR / "accuracy_log.csv"

PREDICTION_START_TAG = "<!-- PREDICTION_TABLE_START -->"
PREDICTION_END_TAG = "<!-- PREDICTION_TABLE_END -->"
ACCURACY_START_TAG = "<!-- ACCURACY_TABLE_START -->"
ACCURACY_END_TAG = "<!-- ACCURACY_TABLE_END -->"


def load_latest_prediction() -> tuple[pd.DataFrame, dict]:
    if not LATEST_PRED_PATH.exists():
        return pd.DataFrame(), {}

    meta = {}
    if LATEST_META_PATH.exists():
        meta = json.loads(LATEST_META_PATH.read_text(encoding="utf-8"))

    return pd.read_csv(LATEST_PRED_PATH), meta


def build_prediction_markdown(prediction: pd.DataFrame, meta: dict) -> str:
    if prediction.empty:
        return "_No latest race prediction has been published yet._\n"

    race_name = meta.get("race_name", "Unknown race")
    race_number = meta.get("race_number", "?")
    generated_at = meta.get("generated_at")

    updated_label = generated_at
    if generated_at:
        try:
            updated_label = datetime.fromisoformat(generated_at).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            updated_label = generated_at

    top = prediction.head(10).copy()

    lines = [
        f"### Latest forecast: {race_name} (Round {race_number})",
        f"_Updated: {updated_label}_\n",
        "| Pos | Driver | Team | Win% | Podium% | Top 5% | DNF% |",
        "|-----|--------|------|------|----------|--------|------|",
    ]

    for _, row in top.iterrows():
        driver = row.get("DriverFull", row.get("Driver", "?"))
        lines.append(
            f"| {int(row.get('PredictedPos', 0))} "
            f"| **{driver}** "
            f"| {row.get('Team', '?')} "
            f"| {row.get('WinProb', 0) * 100:.1f}% "
            f"| {row.get('PodiumProb', 0) * 100:.1f}% "
            f"| {row.get('Top5Prob', 0) * 100:.1f}% "
            f"| {row.get('DNFProb', 0) * 100:.1f}% |"
        )

    lines += [
        "",
        f"Full dashboard: [dashboard/f1_2026_portfolio_dashboard.html](dashboard/f1_2026_portfolio_dashboard.html)",
    ]
    return "\n".join(lines) + "\n"


def build_accuracy_markdown() -> str:
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        comparisons = {
            row["Model"]: row for row in summary.get("baseline_comparison", [])
        }
        ordered = ["ensemble", "quali_baseline", "season_points_baseline"]

        lines = [
            "### Validation benchmark",
            "",
            "| Model | Win accuracy | Spearman rho | MAE positions | Races |",
            "|-------|--------------|--------------|---------------|-------|",
        ]

        for model_name in ordered:
            row = comparisons.get(model_name)
            if not row:
                continue
            label = {
                "ensemble": "Ensemble",
                "quali_baseline": "Qualifying baseline",
                "season_points_baseline": "Season-points baseline",
            }[model_name]
            lines.append(
                f"| {label} "
                f"| {row.get('win_accuracy', 0) * 100:.1f}% "
                f"| {row.get('spearman_rho_mean', 0):.3f} "
                f"| {row.get('mae_positions_mean', 0):.2f} "
                f"| {int(row.get('Races', 0))} |"
            )

        return "\n".join(lines) + "\n"

    if ACCURACY_LOG.exists():
        log = pd.read_csv(ACCURACY_LOG)
        if not log.empty:
            return (
                f"### Live scoring\n\n"
                f"- races scored: {len(log)}\n"
                f"- mean win accuracy: {log.get('win_correct', pd.Series(dtype=float)).mean():.1%}\n"
            )

    return "_No validation summary has been published yet._\n"


def replace_tagged_block(text: str, start_tag: str, end_tag: str, body: str) -> str:
    if start_tag in text and end_tag in text:
        before = text.split(start_tag)[0]
        after = text.split(end_tag)[1]
        return before + start_tag + "\n" + body + end_tag + after

    return text + f"\n\n{start_tag}\n{body}{end_tag}\n"


def update_readme() -> None:
    if not README_PATH.exists():
        raise FileNotFoundError("README.md not found")

    prediction, meta = load_latest_prediction()
    prediction_block = build_prediction_markdown(prediction, meta)
    accuracy_block = build_accuracy_markdown()

    readme = README_PATH.read_text(encoding="utf-8")
    readme = replace_tagged_block(readme, PREDICTION_START_TAG, PREDICTION_END_TAG, prediction_block)
    readme = replace_tagged_block(readme, ACCURACY_START_TAG, ACCURACY_END_TAG, accuracy_block)
    README_PATH.write_text(readme, encoding="utf-8")

    race_name = meta.get("race_name", "unknown race")
    print(f"README refreshed from latest forecast: {race_name}")


if __name__ == "__main__":
    update_readme()
