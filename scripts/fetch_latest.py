"""
scripts/fetch_latest.py

Called by GitHub Actions immediately after a race weekend.
Fetches the latest qualifying and race result data and
saves it to data/processed/ for the prediction pipeline.

This runs BEFORE retrain.py and generate_outputs.py.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, CIRCUITS_2026
from utils.ergast_client import (
    get_race_results,
    get_qualifying_results,
    get_constructor_standings,
    get_driver_standings,
)

import pandas as pd


def get_current_round() -> int:
    """Determine the most recently completed round from stored data."""
    results_path = DATA_PROCESSED / "ergast_results.parquet"
    if not results_path.exists():
        return 0
    results = pd.read_parquet(results_path)
    season = results[results["Year"] == 2026] if "Year" in results.columns else pd.DataFrame()
    return int(season["Round"].max()) if not season.empty else 0


def fetch_latest_race(year: int = 2026) -> bool:
    """
    Fetch the most recently completed race result and qualifying data.
    Appends to the existing parquet files.
    Returns True if new data was fetched.
    """
    completed_round = get_current_round()
    target_round    = completed_round + 1

    # Check if this round exists on the calendar
    circuit_info = next(
        (c for c in CIRCUITS_2026 if c["round"] == target_round), None
    )
    if circuit_info is None:
        print(f"Round {target_round} not found in 2026 calendar — season may be complete")
        return False

    circuit_name = circuit_info["name"]
    print(f"Fetching R{target_round}: {circuit_name} ({year})")

    # Fetch race results
    race_results = get_race_results(year, target_round)
    if race_results.empty:
        print(f"  No race results yet for R{target_round} — race may not have happened")
        return False

    race_results["Circuit"] = circuit_name
    race_results["Year"]    = year

    # Fetch qualifying results
    qual_results = get_qualifying_results(year, target_round)
    if not qual_results.empty:
        qual_results["Circuit"] = circuit_name
        qual_results["Year"]    = year

    # Append to stored results
    results_path = DATA_PROCESSED / "ergast_results.parquet"
    if results_path.exists():
        existing = pd.read_parquet(results_path)
        # Avoid duplicates
        existing = existing[
            ~((existing.get("Year", 0) == year) & (existing.get("Round", 0) == target_round))
        ]
        combined = pd.concat([existing, race_results], ignore_index=True)
    else:
        combined = race_results
    combined.to_parquet(results_path, index=False)
    print(f"  Race results saved: {len(race_results)} rows → {results_path}")

    # Append qualifying
    if not qual_results.empty:
        qual_path = DATA_PROCESSED / "ergast_qualifying.parquet"
        if qual_path.exists():
            existing_qual = pd.read_parquet(qual_path)
            existing_qual = existing_qual[
                ~((existing_qual.get("Year", 0) == year) &
                  (existing_qual.get("Round", 0) == target_round))
            ]
            combined_qual = pd.concat([existing_qual, qual_results], ignore_index=True)
        else:
            combined_qual = qual_results
        combined_qual.to_parquet(qual_path, index=False)
        print(f"  Qualifying saved: {len(qual_results)} rows → {qual_path}")

    # Save this race's qualifying as a named CSV for predict.py to find
    if not qual_results.empty:
        named_qual = DATA_PROCESSED / f"qualifying_2026_r{target_round}.csv"
        qual_results.to_csv(named_qual, index=False)
        print(f"  Named qualifying saved → {named_qual}")

    # Update standings
    standings = get_constructor_standings(year, after_round=target_round)
    if not standings.empty:
        standings["Round"] = target_round
        standings_path = DATA_PROCESSED / "ergast_constructor_standings.parquet"
        if standings_path.exists():
            existing_s = pd.read_parquet(standings_path)
            existing_s = existing_s[
                ~((existing_s.get("Year", 0) == year) &
                  (existing_s.get("Round", 0) == target_round))
            ]
            combined_s = pd.concat([existing_s, standings], ignore_index=True)
        else:
            combined_s = standings
        combined_s.to_parquet(standings_path, index=False)
        print(f"  Constructor standings updated → {standings_path}")

    return True


def fetch_next_qualifying(year: int = 2026) -> bool:
    """
    Fetch qualifying results for the UPCOMING race (Saturday session).
    Called separately when qualifying has just happened.
    """
    completed_round = get_current_round()
    next_round      = completed_round + 1

    circuit_info = next(
        (c for c in CIRCUITS_2026 if c["round"] == next_round), None
    )
    if circuit_info is None:
        return False

    circuit_name = circuit_info["name"]
    print(f"Fetching qualifying for R{next_round}: {circuit_name}")

    qual = get_qualifying_results(year, next_round)
    if qual.empty:
        print(f"  Qualifying not available yet for R{next_round}")
        return False

    qual["Circuit"] = circuit_name
    qual["Year"]    = year

    path = DATA_PROCESSED / f"qualifying_2026_r{next_round}.csv"
    qual.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualifying-only", action="store_true",
                        help="Only fetch qualifying (Saturday)")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if args.qualifying_only:
        success = fetch_next_qualifying(args.year)
    else:
        success = fetch_latest_race(args.year)

    if success:
        print(f"\nData fetch complete.")
    else:
        print(f"\nNo new data fetched — check race calendar / API availability.")