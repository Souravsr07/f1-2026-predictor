"""
Walk-forward backtesting harness for 2022-2025.

Each evaluation season is trained only on seasons strictly before it. This is
stronger than the current single-season backtest entrypoint and is meant to be
the validation harness used when tuning ensemble weights and newer features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BACKTEST_SEASONS, DATA_PROCESSED
from evaluation.metrics import aggregate_metrics
from models.ensemble import EnsemblePredictor
from predict import _get_historical_weather, predict_race
from features.feature_store import build_training_feature_matrix


def _load_parquet(name: str) -> pd.DataFrame:
    path = DATA_PROCESSED / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _build_quali_frame(qual_data: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    race_qual = qual_data[(qual_data["Year"] == year) & (qual_data["Round"] == rnd)].copy()
    if race_qual.empty:
        return pd.DataFrame()
    race_qual["GridPosition"] = race_qual["BestQualTime_s"].rank(method="first").astype(int)
    return race_qual[["Driver", "Team", "BestQualTime_s", "GridPosition"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest harness")
    parser.add_argument(
        "--eval-years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
        help="Evaluation seasons to test in order",
    )
    parser.add_argument("--mc-sims", type=int, default=5000, help="Monte Carlo sims per fitted ensemble")
    args = parser.parse_args()

    master = _load_parquet("master_training_data.parquet")
    standings = _load_parquet("ergast_constructor_standings.parquet")
    fastf1_laps = _load_parquet("fastf1_race_laps.parquet")
    qual_data = _load_parquet("ergast_qualifying.parquet")
    results = _load_parquet("ergast_results.parquet")

    if master.empty or results.empty or qual_data.empty:
        print("Missing processed training files. Run data/pipeline.py first.")
        return

    all_race_metrics: list[dict] = []
    summary_rows: list[dict] = []

    for eval_year in args.eval_years:
        train_years = sorted(year for year in master["Year"].unique() if year < eval_year)
        if not train_years:
            continue

        print(f"\n=== Walk-forward fold: train {train_years} -> eval {eval_year} ===")
        train_master = master[master["Year"].isin(train_years)].copy()
        feature_matrix = build_training_feature_matrix(
            master_data=train_master,
            standings_data=standings,
            fastf1_laps=fastf1_laps,
        )

        ensemble = EnsemblePredictor(mc_n_sims=args.mc_sims)
        ensemble.fit(feature_matrix=feature_matrix, results=train_master, eval_seasons=None)

        season_results = results[results["Year"] == eval_year].copy()
        eval_metrics = []
        for rnd in sorted(season_results["Round"].unique()):
            actual = season_results[season_results["Round"] == rnd].copy()
            circuit = actual["Circuit"].iloc[0]
            quali_df = _build_quali_frame(qual_data, eval_year, rnd)
            if quali_df.empty:
                continue

            pred = predict_race(
                race_name=circuit,
                year=eval_year,
                quali_df=quali_df,
                weather=_get_historical_weather(circuit),
                race_number=int(rnd),
                ensemble=ensemble,
                verbose=False,
            )
            from evaluation.metrics import compute_all_metrics

            metrics = compute_all_metrics(pred, actual[["Driver", "FinishPosition"]])
            metrics["Year"] = eval_year
            metrics["Round"] = int(rnd)
            metrics["Circuit"] = circuit
            eval_metrics.append(metrics)
            all_race_metrics.append(metrics)

        if eval_metrics:
            agg = aggregate_metrics(eval_metrics)
            agg["EvalYear"] = eval_year
            agg["TrainYears"] = ",".join(map(str, train_years))
            summary_rows.append(agg)

    if not all_race_metrics:
        print("No backtest metrics generated.")
        return

    summary_df = pd.DataFrame(summary_rows)
    race_df = pd.DataFrame(all_race_metrics)
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    race_df.to_csv(out_dir / "walk_forward_race_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "walk_forward_summary.csv", index=False)

    overall = aggregate_metrics(all_race_metrics)
    print("\nWalk-forward summary")
    for key, value in overall.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
