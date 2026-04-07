"""
scripts/generate_outputs.py

Runs after every race. Orchestrates:
  1. Load qualifying data for the NEXT race
  2. Generate ensemble prediction
  3. Score the LAST race prediction vs actual results
  4. Save all charts to results/plots/
  5. Append to accuracy log
"""

import sys, os, json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ACTIVE_CIRCUITS_2026, CIRCUITS_2026, DATA_PROCESSED, DRIVER_TEAM_2026
from visualisations import generate_race_week_plots
from championship import ChampionshipForecaster, update_championship_after_race
from evaluation.metrics import compute_all_metrics

RESULTS_DIR   = Path("results")
ACCURACY_LOG  = RESULTS_DIR / "accuracy_log.csv"
STANDINGS_LOG = RESULTS_DIR / "standings.csv"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_2026_results_frame() -> pd.DataFrame:
    live_path = DATA_PROCESSED / "2026_live_results.parquet"
    if live_path.exists():
        return pd.read_parquet(live_path)

    hist_path = DATA_PROCESSED / "ergast_results.parquet"
    if hist_path.exists():
        results = pd.read_parquet(hist_path)
        if "Year" in results.columns:
            return results[results["Year"] == 2026].copy()
        return results

    return pd.DataFrame()


def load_current_race_info() -> dict:
    """Figure out which race is next / just happened from the calendar."""
    results = _load_2026_results_frame()
    if results.empty or "Round" not in results.columns:
        return {"completed_rounds": 0, "next_race": ACTIVE_CIRCUITS_2026[0]}

    completed = int(results["Round"].max())
    next_rnd  = completed + 1
    next_race = next(
        (c for c in ACTIVE_CIRCUITS_2026 if c["round"] == next_rnd),
        ACTIVE_CIRCUITS_2026[-1]
    )
    return {"completed_rounds": completed, "next_race": next_race}


def score_last_race(ensemble, race_info: dict) -> dict:
    """Score the last stored prediction against actual results."""
    completed = race_info["completed_rounds"]
    if completed == 0:
        return {}

    pred_path = RESULTS_DIR / f"prediction_r{completed}.csv"
    if not pred_path.exists():
        print(f"No stored prediction for R{completed} — skipping scoring")
        return {}

    results = _load_2026_results_frame()
    if results.empty:
        return {}

    actual = results[results["Round"] == completed]
    if actual.empty:
        print(f"No actual results found for 2026 R{completed}")
        return {}

    pred   = pd.read_csv(pred_path)
    metrics = compute_all_metrics(pred, actual)
    metrics["Round"]   = completed
    metrics["Year"]    = 2026

    # Append to accuracy log
    metrics_df = pd.DataFrame([metrics])
    if ACCURACY_LOG.exists():
        log = pd.read_csv(ACCURACY_LOG)
        log = pd.concat([log, metrics_df], ignore_index=True)
    else:
        log = metrics_df
    log.to_csv(ACCURACY_LOG, index=False)

    print(f"R{completed} scored: win={'✓' if metrics.get('win_correct') else '✗'}, "
          f"podium={metrics.get('podium_overlap', 0)}/3, "
          f"ρ={metrics.get('spearman_rho', 0):.3f}")
    return metrics


def predict_next_race(ensemble, race_info: dict) -> pd.DataFrame:
    """Generate prediction for the next race."""
    from predict import predict_race, _fetch_weather, _build_default_qualifying
    from dashboard.generate_f1_dashboard import (
        generate_dashboard,
        save_dashboard_archive,
        save_latest_prediction_artifacts,
    )

    next_race = race_info["next_race"]
    race_name = next_race["name"]
    rnd       = next_race["round"]

    print(f"Generating prediction for R{rnd}: {race_name}...")

    # Try to load real qualifying / starting grid for the target round.
    qual_sources = [
        DATA_PROCESSED / f"starting_grid_2026_r{rnd}.csv",
        DATA_PROCESSED / f"qualifying_2026_r{rnd}.csv",
    ]

    quali_df = None
    for qual_path in qual_sources:
        if qual_path.exists():
            quali_df = pd.read_csv(qual_path)
            print(f"  Loaded qualifying from {qual_path}")
            break

    if quali_df is None:
        quali_df = _build_default_qualifying(race_name, year=2026, race_number=rnd)
        print("  Using live proxy qualifying order")

    weather = _fetch_weather(race_name)

    prediction = predict_race(
        race_name   = race_name,
        year        = 2026,
        quali_df    = quali_df,
        weather     = weather,
        race_number = rnd,
        ensemble    = ensemble,
        verbose     = True,
    )

    # Save prediction
    pred_path = RESULTS_DIR / f"prediction_r{rnd}.csv"
    prediction.to_csv(pred_path, index=False)
    print(f"  Prediction saved → {pred_path}")

    # Generate charts
    plots = generate_race_week_plots(
        prediction = prediction,
        race_name  = race_name,
        year       = 2026,
    )
    print(f"  {len(plots)} charts saved")

    dashboard_paths = save_latest_prediction_artifacts(
        prediction=prediction,
        race_name=race_name,
        year=2026,
        race_number=rnd,
    )
    dashboard_path = generate_dashboard()
    dashboard_archive = save_dashboard_archive(
        race_name=race_name,
        year=2026,
        race_number=rnd,
        generated_at=dashboard_paths.get("generated_at"),
    )
    print(f"  Latest prediction snapshot saved -> {dashboard_paths['latest_csv']}")
    print(f"  Prediction archive saved -> {dashboard_paths['archive_csv']}")
    print(f"  Dashboard refreshed -> {dashboard_path}")
    print(f"  Dashboard archive saved -> {dashboard_archive}")

    return prediction


def main():
    from models.ensemble import EnsemblePredictor
    from predict import _load_or_raise_ensemble

    print(f"\n{'='*55}")
    print(f"  F1 2026 Predictor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}\n")

    # Load ensemble
    try:
        ensemble = _load_or_raise_ensemble()
    except FileNotFoundError:
        print("No fitted model found — fitting from scratch...")
        from predict import fit_models
        ensemble = fit_models()

    # Determine race status
    race_info = load_current_race_info()
    print(f"Status: {race_info['completed_rounds']} races complete, "
          f"next: {race_info['next_race']['name']}")

    # Score last race
    score_last_race(ensemble, race_info)

    # Predict next race
    predict_next_race(ensemble, race_info)

    print("\nDone.")


if __name__ == "__main__":
    main()
