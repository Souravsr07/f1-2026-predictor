"""
data/pipeline.py

Master data pipeline. Tries Jolpica API first (Ergast successor),
falls back to OpenF1 API if that fails.

Usage:
    python data/pipeline.py --mode ergast_only   # recommended first run
    python data/pipeline.py --mode merge         # join all sources
    python data/pipeline.py --mode check         # verify data completeness
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRAINING_YEARS, DATA_PROCESSED, SEASON_WEIGHTS
from utils.ergast_client import (
    build_historical_results,
    build_historical_qualifying,
    get_constructor_standings,
)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _normalise_team_name(name: str) -> str:
    mapping = {
        "Red Bull":              "Red Bull",
        "Red Bull Racing":       "Red Bull",
        "McLaren":               "McLaren",
        "Ferrari":               "Ferrari",
        "Mercedes":              "Mercedes",
        "Aston Martin":          "Aston Martin",
        "Alpine F1 Team":        "Alpine",
        "Alpine":                "Alpine",
        "Williams":              "Williams",
        "AlphaTauri":            "Racing Bulls",
        "RB F1 Team":            "Racing Bulls",
        "Racing Bulls":          "Racing Bulls",
        "Haas F1 Team":          "Haas",
        "Haas":                  "Haas",
        "Alfa Romeo":            "Kick Sauber",
        "Sauber":                "Kick Sauber",
        "Kick Sauber":           "Kick Sauber",
        "Toro Rosso":            "Racing Bulls",
        "Racing Point":          "Aston Martin",
        "Force India":           "Aston Martin",
        "Renault":               "Alpine",
    }
    return mapping.get(name, name)


def build_ergast_dataset(years: list[int]) -> bool:
    """
    Pull race results and qualifying from Jolpica API.
    Returns True if data was successfully fetched.
    """
    logger.info("=== Fetching race results from Jolpica API ===")
    logger.info(f"Target years: {years}")
    logger.info("This will take 3-8 minutes. Please wait...")

    results = build_historical_results(years)

    if results.empty:
        logger.error("NO DATA FETCHED — API may be unreachable")
        logger.error("Check your internet connection and try again")
        return False

    results["Team"] = results["Team"].apply(_normalise_team_name)
    out = DATA_PROCESSED / "ergast_results.parquet"
    results.to_parquet(out, index=False)
    logger.info(f"Saved {len(results):,} race result rows → {out}")

    logger.info("=== Fetching qualifying data ===")
    qualifying = build_historical_qualifying(years)
    if not qualifying.empty:
        qualifying["Team"] = qualifying["Team"].apply(_normalise_team_name)
        out = DATA_PROCESSED / "ergast_qualifying.parquet"
        qualifying.to_parquet(out, index=False)
        logger.info(f"Saved {len(qualifying):,} qualifying rows → {out}")

    logger.info("=== Fetching constructor standings ===")
    all_standings = []
    for year in years:
        df = get_constructor_standings(year)
        if not df.empty:
            df["Team"] = df["Team"].apply(_normalise_team_name)
            all_standings.append(df)

    if all_standings:
        standings_df = pd.concat(all_standings, ignore_index=True)
        out = DATA_PROCESSED / "ergast_constructor_standings.parquet"
        standings_df.to_parquet(out, index=False)
        logger.info(f"Saved {len(standings_df):,} standings rows → {out}")

    return True


def build_master_dataset() -> pd.DataFrame:
    """Join all sources into master training DataFrame."""
    logger.info("=== Building master training dataset ===")

    def load(filename):
        path = DATA_PROCESSED / filename
        if path.exists():
            return pd.read_parquet(path)
        logger.warning(f"{filename} not found")
        return pd.DataFrame()

    ergast_results = load("ergast_results.parquet")
    ergast_qual    = load("ergast_qualifying.parquet")
    standings      = load("ergast_constructor_standings.parquet")

    if ergast_results.empty:
        logger.error("No race results — run: python data/pipeline.py --mode ergast_only")
        return pd.DataFrame()

    master = ergast_results.copy()

    if not ergast_qual.empty:
        qual_cols = ["Year", "Round", "Driver", "QualPosition", "BestQualTime_s"]
        available = [c for c in qual_cols if c in ergast_qual.columns]
        master = master.merge(ergast_qual[available], on=["Year", "Round", "Driver"], how="left")

    master["SeasonWeight"] = master["Year"].map(SEASON_WEIGHTS).fillna(0.5)
    master = master.sort_values(["Year", "Round", "FinishPosition"]).reset_index(drop=True)

    logger.info(f"Master dataset: {len(master):,} rows | "
                f"{master['Driver'].nunique()} drivers | "
                f"{master['Year'].nunique()} seasons | "
                f"Years: {sorted(master['Year'].unique())}")

    # Show top active 2026 drivers by weighted score
    from config import DRIVER_TEAM_2026
    active  = set(DRIVER_TEAM_2026.keys())
    recent  = master[master["Year"] >= 2023].copy()
    active_recent = recent[recent["Driver"].isin(active)].copy()
    if not active_recent.empty:
        active_recent["WeightedPos"] = (
            active_recent["FinishPosition"] * active_recent["SeasonWeight"]
        )
        top_drivers = (
            active_recent.groupby("Driver")["WeightedPos"].mean()
            .sort_values().head(10)
        )
        logger.info(f"Top 10 active drivers (2023-2025 weighted):\n{top_drivers.to_string()}")

    out = DATA_PROCESSED / "master_training_data.parquet"
    master.to_parquet(out, index=False)
    logger.info(f"Saved master dataset → {out}")
    return master


def check_data_completeness():
    files = [
        "ergast_results.parquet",
        "ergast_qualifying.parquet",
        "ergast_constructor_standings.parquet",
        "master_training_data.parquet",
    ]
    print("\n=== DATA COMPLETENESS REPORT ===\n")
    all_ok = True
    for f in files:
        path = DATA_PROCESSED / f
        if path.exists():
            df  = pd.read_parquet(path)
            yrs = sorted(df["Year"].unique()) if "Year" in df.columns else []
            print(f"  OK  {f}")
            print(f"       {len(df):,} rows | years: {yrs}")
        else:
            print(f"  MISSING  {f}")
            all_ok = False
        print()

    if all_ok:
        print("All data files present. Run: python predict.py --fit")
    else:
        print("Missing files. Run: python data/pipeline.py --mode ergast_only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 2026 data pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "ergast_only", "merge", "check"],
        default="check",
    )
    parser.add_argument("--years", nargs="+", type=int, default=TRAINING_YEARS)
    args = parser.parse_args()

    if args.mode in ("full", "ergast_only"):
        success = build_ergast_dataset(args.years)
        if success:
            build_master_dataset()
        else:
            logger.error("Pipeline failed — no data fetched")
            sys.exit(1)
    elif args.mode == "merge":
        build_master_dataset()
    elif args.mode == "check":
        check_data_completeness()