"""
predict.py — main entry point for the F1 2026 predictor.

Usage:
    # Predict next race (uses most recent qualifying data)
    python predict.py --race "Bahrain" --year 2026

    # Predict with custom qualifying input
    python predict.py --race "Monaco" --year 2026 --quali-file quali_monaco.csv

    # Run full season forecast (pre-season)
    python predict.py --season 2026

    # Backtest on historical season
    python predict.py --backtest --year 2024

    # Fit models from scratch on all training data
    python predict.py --fit
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import (
    ACTIVE_CIRCUITS_2026,
    CIRCUITS_2026,
    DRIVER_TEAM_2026,
    DATA_PROCESSED,
    BACKTEST_SEASONS,
    TARGET_YEAR,
    TRAINING_YEARS,
)
from features.feature_store import build_prediction_row, MODEL_FEATURES
from models.monte_carlo     import MonteCarloSimulator
from models.ensemble        import EnsemblePredictor, format_prediction_table
from evaluation.metrics     import compute_all_metrics, print_metrics_report


def predict_race(
    race_name: str,
    year: int,
    quali_df: pd.DataFrame = None,
    weather: dict = None,
    race_number: int = None,
    ensemble: EnsemblePredictor = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions for a single race.

    Parameters
    ----------
    race_name   : Circuit name e.g. "Bahrain"
    year        : Season year
    quali_df    : Optional qualifying data. If None, uses dummy grid.
    weather     : Optional weather dict. If None, fetches from API or uses defaults.
    race_number : Round number in the season
    ensemble    : Pre-loaded ensemble (loads from disk if None)
    verbose     : Print prediction table to console

    Returns
    -------
    Prediction DataFrame
    """
    logger.info(f"Predicting {year} {race_name} GP")

    # Load or receive ensemble
    if ensemble is None:
        ensemble = _load_or_raise_ensemble()

    # Get race number from calendar if not provided
    if race_number is None:
        race_number = _get_race_number(race_name, year)

    # Build qualifying data if not provided
    if quali_df is None:
        quali_df = _build_default_qualifying(race_name, year=year, race_number=race_number)

    # Fetch weather if not provided
    if weather is None:
        weather = _fetch_weather(race_name)

    # Build feature rows
    feature_df = build_prediction_row(
        circuit_name = race_name,
        qualifying_df = quali_df,
        weather = weather,
        race_number = race_number,
        year = year,
    )

    # Generate ensemble prediction
    prediction = ensemble.predict(feature_df, race_name=f"{year} {race_name}")

    if verbose:
        print(f"\n{'🏎️  ' * 10}")
        print(format_prediction_table(prediction))
        print(f"\n📊 Win probability distribution:")
        for _, row in prediction.head(10).iterrows():
            bar = "█" * int(row["WinProb"] * 200)
            ci  = f"[{row['WinProb_CI_low']*100:.1f}%–{row['WinProb_CI_high']*100:.1f}%]"
            print(f"  {row['Driver']:<4}  {row['WinProb']*100:5.1f}%  {bar:<30} {ci}")

    return prediction


def run_backtest(
    year: int,
    ensemble: EnsemblePredictor = None,
) -> pd.DataFrame:
    """
    Backtest the model on a historical season.
    For each race: predict using only data available BEFORE that race.

    Returns DataFrame with per-race metrics.
    """
    logger.info(f"Running backtest on {year} season")

    if ensemble is None:
        ensemble = _load_or_raise_ensemble()

    results = _load_parquet("ergast_results.parquet")
    if results.empty:
        raise ValueError("No results data. Run data pipeline first.")

    season_results  = results[results["Year"] == year]
    race_rounds     = sorted(season_results["Round"].unique())

    all_metrics = []
    for rnd in race_rounds:
        race_info = season_results[season_results["Round"] == rnd]
        circuit   = race_info["Circuit"].iloc[0]

        logger.info(f"  Backtesting R{rnd}: {circuit}")

        # Build qualifying from historical data
        qual_data = _load_parquet("ergast_qualifying.parquet")
        race_qual = qual_data[
            (qual_data["Year"] == year) & (qual_data["Round"] == rnd)
        ]
        if race_qual.empty:
            logger.warning(f"  No qualifying data for {year} R{rnd} — skipping")
            continue

        quali_df = race_qual[["Driver", "Team", "BestQualTime_s"]].copy()
        quali_df["GridPosition"] = quali_df["BestQualTime_s"].rank().astype(int)

        # Default weather (historical backtest — no live forecast)
        weather = _get_historical_weather(circuit, year=year, round_number=int(rnd))

        # Predict
        try:
            pred = predict_race(
                race_name   = circuit,
                year        = year,
                quali_df    = quali_df,
                weather     = weather,
                race_number = rnd,
                ensemble    = ensemble,
                verbose     = False,
            )
        except Exception as e:
            logger.error(f"  Prediction failed for {year} R{rnd}: {e}")
            continue

        # Score
        actual = race_info[["Driver", "FinishPosition"]].copy()
        metrics = compute_all_metrics(pred, actual)
        metrics["Year"]    = year
        metrics["Round"]   = rnd
        metrics["Circuit"] = circuit
        all_metrics.append(metrics)

        logger.info(f"  R{rnd} {circuit}: "
                    f"win={'✓' if metrics.get('win_correct') else '✗'}, "
                    f"podium={metrics.get('podium_overlap', 0)}/3, "
                    f"ρ={metrics.get('spearman_rho', 0):.3f}")

    if not all_metrics:
        return pd.DataFrame()

    backtest_df = pd.DataFrame(all_metrics)
    from evaluation.metrics import aggregate_metrics
    agg = aggregate_metrics(all_metrics)

    print(f"\n{'='*50}")
    print(f"  {year} Backtest Summary ({len(all_metrics)} races)")
    print_metrics_report(agg, title=f"{year} Season Backtest")

    return backtest_df


def fit_models(
    training_years: list[int] = None,
    eval_seasons: list[int] = None,
) -> EnsemblePredictor:
    """
    Fit all models from scratch on historical data.
    Saves fitted ensemble to data/processed/.
    """
    from features.feature_store import build_training_feature_matrix

    training_years = training_years or TRAINING_YEARS
    eval_seasons   = eval_seasons   or [2024]

    logger.info(f"Fitting models on {training_years}, eval on {eval_seasons}")

    # Load data
    master_data    = _load_parquet("master_training_data.parquet")
    standings_data = _load_parquet("ergast_constructor_standings.parquet")
    fastf1_laps    = _load_parquet("fastf1_race_laps.parquet")

    if master_data.empty:
        raise ValueError("No training data. Run: python data/pipeline.py --mode full")

    # Build feature matrix
    logger.info("Building feature matrix...")
    feature_matrix = build_training_feature_matrix(
        master_data    = master_data,
        standings_data = standings_data,
        fastf1_laps    = fastf1_laps,
    )

    # Filter to training years
    feature_matrix = feature_matrix[
        feature_matrix["Year"].isin(training_years)
    ]

    # Fit ensemble
    ensemble = EnsemblePredictor(mc_n_sims=10_000)
    ensemble.fit(
        feature_matrix = feature_matrix,
        results        = master_data,
        eval_seasons   = eval_seasons,
    )

    # Save
    ensemble.save()
    logger.info("Ensemble saved. Ready to predict.")
    return ensemble


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_or_raise_ensemble() -> EnsemblePredictor:
    model_path = DATA_PROCESSED / "ensemble_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "No fitted model found. Run: python predict.py --fit"
        )
    return EnsemblePredictor.load(str(model_path))


def _get_race_number(race_name: str, year: int = TARGET_YEAR) -> int:
    for c in _calendar_for_year(year):
        if c["name"].lower() == race_name.lower():
            return c["round"]
    return 1


def _calendar_for_year(year: int) -> list[dict]:
    return ACTIVE_CIRCUITS_2026 if year == TARGET_YEAR else CIRCUITS_2026


def _latest_available_round(frame: pd.DataFrame, race_number: int | None) -> int | None:
    if frame.empty or "Round" not in frame.columns:
        return None

    rounds = sorted(int(rnd) for rnd in frame["Round"].dropna().unique())
    if not rounds:
        return None

    if race_number is None:
        return rounds[-1]

    eligible = [rnd for rnd in rounds if rnd < race_number]
    return max(eligible) if eligible else None


def _build_default_qualifying(
    race_name: str,
    year: int = TARGET_YEAR,
    race_number: int | None = None,
) -> pd.DataFrame:
    """
    Build a fallback qualifying grid.

    For 2026, prefer the most recent live qualifying session as a proxy for the
    next race. If that does not exist, fall back to the configured grid.
    """
    if year == TARGET_YEAR:
        live_qualifying = _load_parquet("2026_live_qualifying.parquet")
        if not live_qualifying.empty and {"Driver", "Team", "GridPosition", "BestQualTime_s"}.issubset(live_qualifying.columns):
            latest_round = _latest_available_round(live_qualifying, race_number)
            proxy = (
                live_qualifying[live_qualifying["Round"] == latest_round].copy()
                if latest_round is not None else pd.DataFrame()
            )
            if not proxy.empty and latest_round is not None:
                proxy = proxy.sort_values(["GridPosition", "Driver"]).reset_index(drop=True)
                logger.info(f"Using live qualifying proxy from 2026 R{latest_round}")
                return proxy[["Driver", "Team", "BestQualTime_s", "GridPosition"]]

        live_results = _load_parquet("2026_live_results.parquet")
        if not live_results.empty and {"Driver", "Team", "FinishPosition"}.issubset(live_results.columns):
            latest_round = _latest_available_round(live_results, race_number)
            proxy = (
                live_results[live_results["Round"] == latest_round].copy()
                if latest_round is not None else pd.DataFrame()
            )
            if not proxy.empty and latest_round is not None:
                proxy = proxy.sort_values(["FinishPosition", "Driver"]).reset_index(drop=True)
                base = 90.0
                proxy["GridPosition"] = np.arange(1, len(proxy) + 1)
                proxy["BestQualTime_s"] = [base + i * 0.15 for i in range(len(proxy))]
                logger.info(f"Using live results proxy from 2026 R{latest_round}")
                return proxy[["Driver", "Team", "BestQualTime_s", "GridPosition"]]

    drivers = list(DRIVER_TEAM_2026.keys())
    teams = [DRIVER_TEAM_2026[d] for d in drivers]
    base = 90.0
    return pd.DataFrame(
        {
            "Driver": drivers,
            "Team": teams,
            "BestQualTime_s": [base + i * 0.15 for i in range(len(drivers))],
            "GridPosition": list(range(1, len(drivers) + 1)),
        }
    )


def _fetch_weather(circuit_name: str) -> dict:
    """Fetch race-day weather (API or fallback)."""
    try:
        from utils.weather_client import get_race_weather_forecast, get_circuit_coords
        coords = get_circuit_coords(circuit_name)
        if coords:
            now = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0)
            return get_race_weather_forecast(circuit_name, now, coords[0], coords[1])
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e} — using defaults")

    from utils.weather_client import get_historical_race_weather
    return get_historical_race_weather(circuit_name)


def _get_historical_weather(
    circuit_name: str,
    year: int | None = None,
    round_number: int | None = None,
) -> dict:
    """Historical average weather for backtesting."""
    from utils.weather_client import get_historical_race_weather
    return get_historical_race_weather(circuit_name, year=year, round_number=round_number)


def _load_parquet(filename: str) -> pd.DataFrame:
    path = DATA_PROCESSED / filename
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _publish_prediction_artifacts(
    prediction: pd.DataFrame,
    race_name: str,
    year: int,
    race_number: int | None = None,
) -> dict[str, Path]:
    """
    Save the latest race prediction and refresh the portfolio dashboard.
    """
    try:
        from dashboard.generate_f1_dashboard import (
            generate_dashboard,
            save_dashboard_archive,
            save_latest_prediction_artifacts,
        )

        paths = save_latest_prediction_artifacts(
            prediction=prediction,
            race_name=race_name,
            year=year,
            race_number=race_number,
        )
        dashboard_path = generate_dashboard()
        dashboard_archive = save_dashboard_archive(
            race_name=race_name,
            year=year,
            race_number=race_number,
            generated_at=paths.get("generated_at"),
        )
        logger.info(f"Latest prediction saved -> {paths['latest_csv']}")
        logger.info(f"Prediction archive saved -> {paths['archive_csv']}")
        logger.info(f"Dashboard refreshed -> {dashboard_path}")
        logger.info(f"Dashboard archive saved -> {dashboard_archive}")
        paths["dashboard"] = dashboard_path
        paths["dashboard_archive"] = dashboard_archive
        return paths
    except Exception as exc:
        logger.warning(f"Could not refresh dashboard artifacts: {exc}")
        return {}


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="F1 2026 Race Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --fit
  python predict.py --race Bahrain --year 2026
  python predict.py --backtest --year 2024
        """
    )

    parser.add_argument("--race",      type=str,  help="Circuit name to predict")
    parser.add_argument("--year",      type=int,  default=2026, help="Season year")
    parser.add_argument("--round",     type=int,  help="Round number (overrides calendar lookup)")
    parser.add_argument("--quali-file",type=str,  help="Path to CSV with qualifying times")
    parser.add_argument("--fit",       action="store_true", help="Fit models from scratch")
    parser.add_argument("--backtest",  action="store_true", help="Run backtest on --year season")
    parser.add_argument("--season",    action="store_true", help="Full season forecast")
    parser.add_argument("--components",action="store_true", help="Show individual model outputs")
    args = parser.parse_args()

    if args.fit:
        ensemble = fit_models()

    elif args.backtest:
        run_backtest(year=args.year)

    elif args.race:
        quali_df = None
        if args.quali_file:
            quali_df = pd.read_csv(args.quali_file)
            logger.info(f"Loaded qualifying data from {args.quali_file}")

        race_number = args.round if args.round is not None else _get_race_number(args.race, args.year)
        prediction = predict_race(
            race_name   = args.race,
            year        = args.year,
            quali_df    = quali_df,
            race_number = race_number,
        )
        _publish_prediction_artifacts(
            prediction=prediction,
            race_name=args.race,
            year=args.year,
            race_number=race_number,
        )

    elif args.season:
        logger.info(f"Running full {args.year} season forecast...")
        ensemble = _load_or_raise_ensemble()
        all_preds = []
        for circuit_info in _calendar_for_year(args.year):
            try:
                pred = predict_race(
                    race_name   = circuit_info["name"],
                    year        = args.year,
                    race_number = circuit_info["round"],
                    ensemble    = ensemble,
                    verbose     = False,
                )
                pred["Circuit"] = circuit_info["name"]
                pred["Round"]   = circuit_info["round"]
                all_preds.append(pred)
            except Exception as e:
                logger.error(f"Failed {circuit_info['name']}: {e}")
                continue

        if all_preds:
            season_df = pd.concat(all_preds, ignore_index=True)
            out = DATA_PROCESSED / f"season_forecast_{args.year}.parquet"
            season_df.to_parquet(out)
            logger.info(f"Season forecast saved → {out}")

            # Print championship forecast
            win_counts = season_df.groupby("Driver")["WinProb"].sum().sort_values(ascending=False)
            print("\n🏆 Projected Championship Win Probabilities (season total)")
            for driver, prob in win_counts.head(10).items():
                print(f"  {driver:<4}  {prob:.2f} expected wins")

    else:
        parser.print_help()
