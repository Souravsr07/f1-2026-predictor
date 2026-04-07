"""
Comprehensive diagnostics for the F1 predictor.

This script is meant to answer two questions:
1. How accurate is the model versus simple baselines?
2. Where is the model unreliable or internally inconsistent?

Outputs in results/diagnostics/:
  - baseline_comparison.csv
  - walk_forward_race_metrics.csv
  - walk_forward_year_metrics.csv
  - circuit_breakdown.csv
  - team_residuals.csv
  - calibration_win.csv
  - calibration_podium.csv
  - sanity_report.csv
  - live_accuracy_tracker.csv
  - baseline_comparison.png
  - calibration.png
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import features.feature_store as feature_store
from config import ACTIVE_CIRCUITS_2026, CIRCUITS_2026, DATA_PROCESSED, TARGET_YEAR
from evaluation.metrics import aggregate_metrics, compute_all_metrics
from models.ensemble import EnsemblePredictor
from predict import _get_historical_weather, predict_race
from utils.name_normalization import normalize_race_name


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_parquet(name: str) -> pd.DataFrame:
    path = DATA_PROCESSED / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@contextmanager
def _temporary_feature_store_output() -> Path:
    """
    build_training_feature_matrix() always writes a parquet. During diagnostics we
    keep that write isolated so the script is low-risk to run repeatedly.
    """
    original = feature_store.DATA_PROCESSED
    with tempfile.TemporaryDirectory(prefix="f1_diagnostics_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        feature_store.DATA_PROCESSED = tmp_path
        try:
            yield tmp_path
        finally:
            feature_store.DATA_PROCESSED = original


def _build_training_matrix(
    master_data: pd.DataFrame,
    standings_data: pd.DataFrame,
    fastf1_laps: pd.DataFrame,
) -> pd.DataFrame:
    with _temporary_feature_store_output():
        return feature_store.build_training_feature_matrix(
            master_data=master_data,
            standings_data=standings_data,
            fastf1_laps=fastf1_laps,
        )


def _build_quali_frame(qual_data: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    race_qual = qual_data[(qual_data["Year"] == year) & (qual_data["Round"] == rnd)].copy()
    if race_qual.empty or "BestQualTime_s" not in race_qual.columns:
        return pd.DataFrame()

    race_qual = race_qual.dropna(subset=["BestQualTime_s"])
    if len(race_qual) < 10:
        return pd.DataFrame()

    race_qual["GridPosition"] = race_qual["BestQualTime_s"].rank(method="first").astype(int)
    return race_qual[["Driver", "Team", "BestQualTime_s", "GridPosition"]].copy()


def _build_qualifying_baseline(quali_df: pd.DataFrame) -> pd.DataFrame:
    baseline = quali_df[["Driver", "Team", "GridPosition"]].copy()
    baseline = baseline.sort_values(["GridPosition", "Driver"]).reset_index(drop=True)
    baseline["PredictedPos"] = baseline.index + 1
    return baseline


def _build_points_baseline(
    quali_df: pd.DataFrame,
    season_results: pd.DataFrame,
    rnd: int,
) -> pd.DataFrame:
    baseline = quali_df[["Driver", "Team", "GridPosition"]].copy()
    prior_points = (
        season_results[season_results["Round"] < rnd]
        .groupby("Driver")["Points"]
        .sum()
        .rename("SeasonPointsSoFar")
    )
    baseline = baseline.merge(prior_points, on="Driver", how="left")
    baseline["SeasonPointsSoFar"] = baseline["SeasonPointsSoFar"].fillna(0.0)
    baseline = baseline.sort_values(
        ["SeasonPointsSoFar", "GridPosition"],
        ascending=[False, True],
    ).reset_index(drop=True)
    baseline["PredictedPos"] = baseline.index + 1
    return baseline


def _prepare_metric_row(metrics: dict, model_name: str, year: int, rnd: int, circuit: str) -> dict:
    row = dict(metrics)
    row["Model"] = model_name
    row["Year"] = int(year)
    row["Round"] = int(rnd)
    row["Circuit"] = circuit
    return row


def _metric_records(frame: pd.DataFrame) -> list[dict]:
    metric_cols = [
        "win_correct",
        "podium_overlap",
        "podium_overlap_pct",
        "spearman_rho",
        "spearman_pval",
        "mae_positions",
        "brier_win",
        "brier_podium",
    ]
    available = [col for col in metric_cols if col in frame.columns]
    if not available:
        return []
    return frame[available].to_dict("records")


def _calibration_table(
    predictions: pd.DataFrame,
    prob_col: str,
    actual_col: str,
    n_bins: int = 8,
) -> pd.DataFrame:
    if predictions.empty or prob_col not in predictions.columns or actual_col not in predictions.columns:
        return pd.DataFrame()

    frame = predictions[[prob_col, actual_col]].dropna().copy()
    if frame.empty:
        return pd.DataFrame()

    frame["bin"] = pd.cut(
        frame[prob_col],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
    )
    table = (
        frame.groupby("bin", observed=False)
        .agg(
            count=(prob_col, "size"),
            predicted_mean=(prob_col, "mean"),
            actual_rate=(actual_col, "mean"),
        )
        .reset_index()
    )
    table["gap"] = table["actual_rate"] - table["predicted_mean"]
    return table


def _sanity_checks(prediction: pd.DataFrame, race_name: str, year: int, rnd: int) -> pd.DataFrame:
    checks: list[dict] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "race": race_name,
                "year": year,
                "round": rnd,
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    if prediction.empty:
        add_check("prediction_not_empty", False, "prediction dataframe is empty")
        return pd.DataFrame(checks)

    win_sum = float(prediction["WinProb"].sum()) if "WinProb" in prediction.columns else np.nan
    podium_sum = float(prediction["PodiumProb"].sum()) if "PodiumProb" in prediction.columns else np.nan
    dnf_unique = int(prediction["DNFProb"].nunique()) if "DNFProb" in prediction.columns else 0

    add_check("win_prob_sums_to_one", abs(win_sum - 1.0) < 0.02, f"sum={win_sum:.4f}")
    add_check("podium_prob_sums_to_three", abs(podium_sum - 3.0) < 0.08, f"sum={podium_sum:.4f}")
    add_check("predicted_positions_unique", prediction["PredictedPos"].is_unique, "positions unique")
    add_check("win_prob_monotonic", prediction["WinProb"].is_monotonic_decreasing, "sorted by WinProb")
    add_check(
        "ci_ordered",
        bool((prediction["WinProb_CI_low"] <= prediction["WinProb_CI_high"]).all()),
        "low <= high for all rows",
    )
    add_check(
        "ci_contains_win_prob",
        bool(
            (
                (prediction["WinProb"] >= prediction["WinProb_CI_low"])
                & (prediction["WinProb"] <= prediction["WinProb_CI_high"])
            ).all()
        ),
        "checks whether displayed WinProb lies inside printed CI",
    )
    add_check("dnf_prob_not_constant", dnf_unique > 1, f"unique_values={dnf_unique}")
    add_check(
        "dnf_prob_nonzero_exists",
        bool((prediction["DNFProb"] > 0).any()) if "DNFProb" in prediction.columns else False,
        f"max_dnf={prediction['DNFProb'].max():.4f}" if "DNFProb" in prediction.columns else "DNFProb missing",
    )
    add_check(
        "probabilities_non_negative",
        bool((prediction[["WinProb", "PodiumProb", "Top10Prob", "DNFProb"]] >= 0).all().all()),
        "all displayed probabilities >= 0",
    )

    return pd.DataFrame(checks)


def _build_live_accuracy_tracker() -> pd.DataFrame:
    live_results = _load_parquet("2026_live_results.parquet")
    if live_results.empty:
        return pd.DataFrame()

    rows = []
    root_results = Path(__file__).resolve().parent.parent / "results"
    for pred_path in sorted(root_results.glob("prediction_r*.csv")):
        round_text = pred_path.stem.replace("prediction_r", "")
        if not round_text.isdigit():
            continue
        rnd = int(round_text)
        actual = live_results[live_results["Round"] == rnd][["Driver", "FinishPosition"]].copy()
        if actual.empty:
            continue
        pred = pd.read_csv(pred_path)
        metrics = compute_all_metrics(pred, actual)
        if not metrics:
            continue
        metrics["Round"] = rnd
        rows.append(metrics)

    return pd.DataFrame(rows)


def _save_baseline_plot(baseline_summary: pd.DataFrame) -> None:
    if baseline_summary.empty:
        return

    metrics = ["win_accuracy", "spearman_rho_mean", "mae_positions_mean"]
    available = [metric for metric in metrics if metric in baseline_summary.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        values = baseline_summary.set_index("Model")[metric].sort_values(ascending=(metric == "mae_positions_mean"))
        ax.bar(values.index, values.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(values)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
        if metric == "mae_positions_mean":
            ax.set_ylabel("lower is better")
        else:
            ax.set_ylabel("higher is better")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _save_calibration_plot(win_cal: pd.DataFrame, podium_cal: pd.DataFrame) -> None:
    if win_cal.empty and podium_cal.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    diag = np.linspace(0, 1, 100)

    for ax, table, title in zip(
        axes,
        [win_cal, podium_cal],
        ["Win calibration", "Podium calibration"],
    ):
        ax.plot(diag, diag, linestyle="--", color="#777777", linewidth=1)
        if not table.empty:
            ax.plot(table["predicted_mean"], table["actual_rate"], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "calibration.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model diagnostics and reliability checks")
    parser.add_argument("--eval-years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--mc-sims", type=int, default=300, help="Monte Carlo sims per walk-forward fold")
    parser.add_argument("--sample-race", type=str, default=None, help="Optional race name for sanity-check prediction")
    parser.add_argument("--sample-round", type=int, default=None, help="Optional round for sanity-check prediction")
    parser.add_argument("--sample-year", type=int, default=TARGET_YEAR, help="Year for sanity-check prediction")
    args = parser.parse_args()

    master = _load_parquet("master_training_data.parquet")
    standings = _load_parquet("ergast_constructor_standings.parquet")
    fastf1_laps = _load_parquet("fastf1_race_laps.parquet")
    qual_data = _load_parquet("ergast_qualifying.parquet")
    results = _load_parquet("ergast_results.parquet")

    if master.empty or standings.empty or qual_data.empty or results.empty:
        raise ValueError("Missing processed training inputs. Run the data pipeline first.")

    all_model_metrics: list[dict] = []
    all_grid_metrics: list[dict] = []
    all_points_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    all_actuals: list[pd.DataFrame] = []

    year_summaries: list[dict] = []

    for eval_year in args.eval_years:
        train_years = sorted(int(year) for year in master["Year"].unique() if year < eval_year)
        if not train_years:
            continue

        train_master = master[master["Year"].isin(train_years)].copy()
        feature_matrix = _build_training_matrix(train_master, standings, fastf1_laps)
        ensemble = EnsemblePredictor(mc_n_sims=args.mc_sims)
        ensemble.fit(feature_matrix=feature_matrix, results=train_master, eval_seasons=None)

        season_results = results[results["Year"] == eval_year].copy()
        season_model_metrics: list[dict] = []
        season_grid_metrics: list[dict] = []
        season_points_metrics: list[dict] = []

        for rnd in sorted(season_results["Round"].unique()):
            actual = season_results[season_results["Round"] == rnd].copy()
            if actual.empty:
                continue

            circuit_raw = actual["Circuit"].iloc[0]
            circuit = normalize_race_name(circuit_raw) or circuit_raw
            quali_df = _build_quali_frame(qual_data, eval_year, int(rnd))
            if quali_df.empty:
                continue

            try:
                pred = predict_race(
                    race_name=circuit,
                    year=eval_year,
                    quali_df=quali_df,
                    weather=_get_historical_weather(circuit, year=eval_year, round_number=int(rnd)),
                    race_number=int(rnd),
                    ensemble=ensemble,
                    verbose=False,
                )
            except Exception as exc:
                print(f"SKIP {eval_year} R{rnd} {circuit}: {exc}")
                continue

            actual_small = actual[["Driver", "Team", "FinishPosition"]].copy()
            pred_small = pred.copy()
            pred_small["Year"] = eval_year
            pred_small["Round"] = int(rnd)
            pred_small["Circuit"] = circuit
            all_predictions.append(pred_small)
            all_actuals.append(actual_small.assign(Year=eval_year, Round=int(rnd), Circuit=circuit))

            model_metrics = _prepare_metric_row(
                compute_all_metrics(pred_small, actual_small),
                "ensemble",
                eval_year,
                int(rnd),
                circuit,
            )
            grid_metrics = _prepare_metric_row(
                compute_all_metrics(_build_qualifying_baseline(quali_df), actual_small),
                "quali_baseline",
                eval_year,
                int(rnd),
                circuit,
            )
            points_metrics = _prepare_metric_row(
                compute_all_metrics(_build_points_baseline(quali_df, season_results, int(rnd)), actual_small),
                "season_points_baseline",
                eval_year,
                int(rnd),
                circuit,
            )

            all_model_metrics.append(model_metrics)
            all_grid_metrics.append(grid_metrics)
            all_points_metrics.append(points_metrics)
            season_model_metrics.append(model_metrics)
            season_grid_metrics.append(grid_metrics)
            season_points_metrics.append(points_metrics)

        if season_model_metrics:
            ensemble_year = aggregate_metrics(season_model_metrics)
            ensemble_year["Model"] = "ensemble"
            ensemble_year["EvalYear"] = eval_year
            ensemble_year["Races"] = len(season_model_metrics)
            year_summaries.append(ensemble_year)

            grid_year = aggregate_metrics(season_grid_metrics)
            grid_year["Model"] = "quali_baseline"
            grid_year["EvalYear"] = eval_year
            grid_year["Races"] = len(season_grid_metrics)
            year_summaries.append(grid_year)

            points_year = aggregate_metrics(season_points_metrics)
            points_year["Model"] = "season_points_baseline"
            points_year["EvalYear"] = eval_year
            points_year["Races"] = len(season_points_metrics)
            year_summaries.append(points_year)

    model_race_df = pd.DataFrame(all_model_metrics)
    grid_race_df = pd.DataFrame(all_grid_metrics)
    points_race_df = pd.DataFrame(all_points_metrics)
    all_race_metrics = pd.concat([model_race_df, grid_race_df, points_race_df], ignore_index=True)

    baseline_rows = []
    for name, frame in [
        ("ensemble", model_race_df),
        ("quali_baseline", grid_race_df),
        ("season_points_baseline", points_race_df),
    ]:
        if frame.empty:
            continue
        agg = aggregate_metrics(_metric_records(frame))
        agg["Model"] = name
        agg["Races"] = len(frame)
        baseline_rows.append(agg)
    baseline_summary = pd.DataFrame(baseline_rows)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    actuals_df = pd.concat(all_actuals, ignore_index=True) if all_actuals else pd.DataFrame()
    merged_eval = predictions_df.merge(
        actuals_df[["Driver", "Year", "Round", "Circuit", "FinishPosition", "Team"]],
        on=["Driver", "Year", "Round", "Circuit"],
        how="inner",
        suffixes=("", "_actual"),
    ) if not predictions_df.empty and not actuals_df.empty else pd.DataFrame()

    circuit_breakdown = pd.DataFrame()
    team_residuals = pd.DataFrame()
    win_cal = pd.DataFrame()
    podium_cal = pd.DataFrame()
    if not merged_eval.empty:
        circuit_rows = []
        for circuit, group in model_race_df.groupby("Circuit"):
            metrics = aggregate_metrics(_metric_records(group))
            metrics["Circuit"] = circuit
            metrics["Races"] = int(group["Round"].nunique()) if "Round" in group.columns else len(group)
            circuit_rows.append(metrics)
        circuit_breakdown = pd.DataFrame(circuit_rows).sort_values(
            ["spearman_rho_mean", "Races"],
            ascending=[False, False],
        )

        merged_eval["pos_error"] = merged_eval["PredictedPos"] - merged_eval["FinishPosition"]
        merged_eval["abs_pos_error"] = merged_eval["pos_error"].abs()
        team_residuals = (
            merged_eval.groupby("Team", as_index=False)
            .agg(
                races=("Round", "nunique"),
                mean_pos_error=("pos_error", "mean"),
                mae=("abs_pos_error", "mean"),
                mean_win_prob=("WinProb", "mean"),
                actual_wins=("FinishPosition", lambda values: int((values == 1).sum())),
            )
            .sort_values("mae")
        )

        merged_eval["ActualWin"] = (merged_eval["FinishPosition"] == 1).astype(float)
        merged_eval["ActualPodium"] = (merged_eval["FinishPosition"] <= 3).astype(float)
        win_cal = _calibration_table(merged_eval, "WinProb", "ActualWin")
        podium_cal = _calibration_table(merged_eval, "PodiumProb", "ActualPodium")

    sample_year = args.sample_year
    sample_round = args.sample_round
    sample_race = args.sample_race

    if sample_race is None and sample_year == TARGET_YEAR:
        live_results = _load_parquet("2026_live_results.parquet")
        completed_rounds = int(live_results["Round"].max()) if not live_results.empty else 0
        sample_round = sample_round or completed_rounds + 1
        sample_race = sample_race or next(
            (item["name"] for item in ACTIVE_CIRCUITS_2026 if item["round"] == sample_round),
            ACTIVE_CIRCUITS_2026[0]["name"],
        )

    sanity_report = pd.DataFrame()
    if sample_race is not None and sample_round is not None:
        sample_prediction = predict_race(
            race_name=sample_race,
            year=sample_year,
            race_number=sample_round,
            verbose=False,
        )
        sanity_report = _sanity_checks(sample_prediction, sample_race, sample_year, sample_round)
        sample_prediction.to_csv(RESULTS_DIR / "sample_prediction.csv", index=False)

    live_accuracy_tracker = _build_live_accuracy_tracker()

    baseline_summary.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)
    all_race_metrics.to_csv(RESULTS_DIR / "walk_forward_race_metrics.csv", index=False)
    pd.DataFrame(year_summaries).to_csv(RESULTS_DIR / "walk_forward_year_metrics.csv", index=False)
    circuit_breakdown.to_csv(RESULTS_DIR / "circuit_breakdown.csv", index=False)
    team_residuals.to_csv(RESULTS_DIR / "team_residuals.csv", index=False)
    win_cal.to_csv(RESULTS_DIR / "calibration_win.csv", index=False)
    podium_cal.to_csv(RESULTS_DIR / "calibration_podium.csv", index=False)
    sanity_report.to_csv(RESULTS_DIR / "sanity_report.csv", index=False)
    live_accuracy_tracker.to_csv(RESULTS_DIR / "live_accuracy_tracker.csv", index=False)

    summary_payload = {
        "baseline_comparison": baseline_rows,
        "files_written": sorted(path.name for path in RESULTS_DIR.glob("*")),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _save_baseline_plot(baseline_summary)
    _save_calibration_plot(win_cal, podium_cal)

    print("\nDiagnostics summary")
    if not baseline_summary.empty:
        print(baseline_summary[["Model", "Races", "win_accuracy", "podium_overlap_mean", "spearman_rho_mean", "mae_positions_mean"]].to_string(index=False))
    if not sanity_report.empty:
        print("\nSanity checks")
        print(sanity_report[["check", "status", "details"]].to_string(index=False))
    print(f"\nOutputs saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
