"""
Run a lightweight EDA pass for the 2026 upgrade.

Outputs are written to results/eda/ so they can be inspected without a
notebook:
  - feature_target_correlation.csv
  - yearly_feature_drift.csv
  - live_2026_team_shift.csv
  - driver_vs_car_signal.csv
  - leakage_report.csv
  - correlation_bar.png
  - team_shift.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED
from scripts.ingest_live_2026_data import ingest_live_data


EDA_DIR = Path(__file__).resolve().parent.parent / "results" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_ingest_live() -> dict[str, pd.DataFrame]:
    results_path = DATA_PROCESSED / "2026_live_results.parquet"
    if results_path.exists():
        payload = {
            "results": pd.read_parquet(results_path),
            "qualifying": pd.read_parquet(DATA_PROCESSED / "2026_live_qualifying.parquet")
            if (DATA_PROCESSED / "2026_live_qualifying.parquet").exists()
            else pd.DataFrame(),
            "sprint": pd.read_parquet(DATA_PROCESSED / "2026_live_sprint.parquet")
            if (DATA_PROCESSED / "2026_live_sprint.parquet").exists()
            else pd.DataFrame(),
            "pace": pd.read_parquet(DATA_PROCESSED / "2026_live_pace.parquet")
            if (DATA_PROCESSED / "2026_live_pace.parquet").exists()
            else pd.DataFrame(),
        }
        return payload

    ingested = ingest_live_data()
    return {
        "results": ingested.get("2026_live_results", pd.DataFrame()),
        "qualifying": ingested.get("2026_live_qualifying", pd.DataFrame()),
        "sprint": ingested.get("2026_live_sprint", pd.DataFrame()),
        "pace": ingested.get("2026_live_pace", pd.DataFrame()),
    }


def _plot_correlations(corr_df: pd.DataFrame) -> None:
    top = pd.concat([corr_df.head(8), corr_df.tail(8)]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E8002D" if value > 0 else "#27F4D2" for value in top["corr_with_finish"]]
    ax.barh(top["feature"], top["corr_with_finish"], color=colors, alpha=0.88)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Correlation with finish position")
    ax.set_title("Historical feature correlation with finish position")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "correlation_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_team_shift(team_shift: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(team_shift["Team"], team_shift["PointsShift"], color="#3671C6", alpha=0.88)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Live 2026 avg race-points shift vs 2025")
    ax.set_title("Team form drift: 2026 live vs 2025 baseline")
    for bar, (_, row) in zip(bars, team_shift.iterrows()):
        ax.text(
            bar.get_width() + (0.05 if bar.get_width() >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{row['PointsShift']:+.2f}",
            va="center",
            ha="left" if bar.get_width() >= 0 else "right",
        )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "team_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    feature_matrix_path = DATA_PROCESSED / "feature_matrix.parquet"
    master_path = DATA_PROCESSED / "master_training_data.parquet"
    if not feature_matrix_path.exists() or not master_path.exists():
        print("Missing processed training data. Run data/pipeline.py and predict.py --fit first.")
        return

    feature_matrix = pd.read_parquet(feature_matrix_path)
    master = pd.read_parquet(master_path)
    live = _load_or_ingest_live()

    numeric_cols = feature_matrix.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"Year", "Round", "FinishPosition", "SeasonWeight"}]

    corr_rows = []
    for col in feature_cols:
        series = feature_matrix[col]
        if series.nunique(dropna=True) <= 1:
            corr = 0.0
        else:
            corr = float(series.corr(feature_matrix["FinishPosition"], method="spearman"))
        corr_rows.append({"feature": col, "corr_with_finish": corr})
    corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_finish", ascending=False).reset_index(drop=True)
    corr_df.to_csv(EDA_DIR / "feature_target_correlation.csv", index=False)
    _plot_correlations(corr_df)

    drift_features = [
        col
        for col in [
            "GapToPole_s",
            "GridPositionNorm",
            "RollingAvgFinish_5",
            "DiscountedConstructorScore",
            "WeatherRiskScore",
            "CarMomentum_5race",
            "CarMomentumDelta",
            "TeamQualiGap_s",
        ]
        if col in feature_matrix.columns
    ]
    yearly_drift = (
        feature_matrix.groupby("Year")[drift_features]
        .mean()
        .reset_index()
    )
    yearly_drift.to_csv(EDA_DIR / "yearly_feature_drift.csv", index=False)

    live_results = live["results"]
    if not live_results.empty:
        hist_2025 = master[master["Year"] == 2025].copy()
        hist_2025_team = (
            hist_2025.groupby("Team", as_index=False)["Points"]
            .mean()
            .rename(columns={"Points": "Hist2025AvgRacePoints"})
        )
        live_team = (
            live_results.groupby("Team", as_index=False)["Points"]
            .mean()
            .rename(columns={"Points": "Live2026AvgRacePoints"})
        )
        team_shift = live_team.merge(hist_2025_team, on="Team", how="outer").fillna(0.0)
        team_shift["PointsShift"] = team_shift["Live2026AvgRacePoints"] - team_shift["Hist2025AvgRacePoints"]
        team_shift = team_shift.sort_values("PointsShift", ascending=False).reset_index(drop=True)
        team_shift.to_csv(EDA_DIR / "live_2026_team_shift.csv", index=False)
        _plot_team_shift(team_shift)

        live_qual = live["qualifying"]
        merged_live = live_results.merge(
            live_qual[["Round", "Circuit", "Driver", "QualPosition", "BestQualTime_s"]],
            on=["Round", "Circuit", "Driver"],
            how="left",
        )
        team_avg_points = live_results.groupby("Team")["Points"].mean().rename("TeamAvgRacePoints")
        merged_live = merged_live.merge(team_avg_points.reset_index(), on="Team", how="left")

        signal_rows = []
        signal_defs = {
            "grid_vs_finish": ("GridPosition", "FinishPosition"),
            "quali_time_vs_finish": ("BestQualTime_s", "FinishPosition"),
            "team_avg_points_vs_finish": ("TeamAvgRacePoints", "FinishPosition"),
            "constructor_live_points_vs_finish": ("Points", "FinishPosition"),
        }
        for label, (left, right) in signal_defs.items():
            if left in merged_live.columns and merged_live[left].notna().sum() > 2:
                corr = float(merged_live[left].corr(merged_live[right], method="spearman"))
                signal_rows.append({"signal": label, "spearman_corr": corr})
        pd.DataFrame(signal_rows).to_csv(EDA_DIR / "driver_vs_car_signal.csv", index=False)
    else:
        pd.DataFrame(columns=["signal", "spearman_corr"]).to_csv(EDA_DIR / "driver_vs_car_signal.csv", index=False)
        pd.DataFrame(columns=["Team", "Live2026AvgRacePoints", "Hist2025AvgRacePoints", "PointsShift"]).to_csv(
            EDA_DIR / "live_2026_team_shift.csv",
            index=False,
        )

    leakage_checks = []
    constant_suspects = [
        "WetPerformanceRating",
        "WeatherRiskScore",
        "RainProbability",
        "Temperature_c",
        "IsWetRace",
        "CarMomentum_5race",
        "CarMomentumDelta",
        "TeamQualiGap_s",
    ]
    for col in constant_suspects:
        if col in feature_matrix.columns:
            leakage_checks.append(
                {
                    "check": f"{col} not constant",
                    "status": "FAIL" if feature_matrix[col].nunique(dropna=False) <= 1 else "PASS",
                    "details": f"unique_values={feature_matrix[col].nunique(dropna=False)}",
                }
            )

    leakage_checks.extend(
        [
            {
                "check": "Constructor reference year leakage risk",
                "status": "WARN",
                "details": "constructor.py currently anchors scores to a fixed reference year for all training rows",
            },
            {
                "check": "Circuit type form leakage risk",
                "status": "WARN",
                "details": "driver_form.py computes circuit-type averages without an explicit prior-race shift",
            },
            {
                "check": "Wet rating leakage risk",
                "status": "WARN",
                "details": "wet ratings are computed from global wet/dry splits instead of strict walk-forward slices",
            },
        ]
    )
    pd.DataFrame(leakage_checks).to_csv(EDA_DIR / "leakage_report.csv", index=False)

    print(f"EDA outputs written to {EDA_DIR}")


if __name__ == "__main__":
    main()

