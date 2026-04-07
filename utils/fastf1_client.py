"""
utils/fastf1_client.py

Clean wrapper around the FastF1 API. Handles caching, retries, and
standardises all returned DataFrames to consistent column names and dtypes.
All raw data fetching goes through this module — nothing else imports fastf1
directly.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FASTF1_CACHE, TRAINING_YEARS

# Suppress fastf1's noisy deprecation warnings during load
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable cache once at import time
FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE))


# ── Session loader ─────────────────────────────────────────────────────────

def load_session(
    year: int,
    event: str | int,
    session_type: str = "R",
    retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[fastf1.core.Session]:
    """
    Load a FastF1 session with retry logic.

    Parameters
    ----------
    year        : Season year
    event       : Round number (int) or event name (str) e.g. "Bahrain"
    session_type: "R" race, "Q" qualifying, "FP1"/"FP2"/"FP3" practice
    retries     : Number of attempts before giving up
    retry_delay : Seconds to wait between retries

    Returns
    -------
    fastf1.core.Session or None if all retries fail
    """
    for attempt in range(1, retries + 1):
        try:
            session = fastf1.get_session(year, event, session_type)
            session.load(
                laps=True,
                telemetry=False,   # telemetry is slow; enable per-request if needed
                weather=True,
                messages=False,
            )
            logger.info(f"Loaded {year} {event} {session_type} (attempt {attempt})")
            return session
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{retries} failed for {year} {event} {session_type}: {e}")
            if attempt < retries:
                time.sleep(retry_delay)
    logger.error(f"All retries exhausted for {year} {event} {session_type}")
    return None


# ── Lap time extraction ────────────────────────────────────────────────────

def get_race_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract clean lap data from a race session.

    Returns a DataFrame with one row per lap, columns:
        Driver, LapNumber, LapTime_s, Sector1_s, Sector2_s, Sector3_s,
        Compound, TyreLife, IsPersonalBest, TrackStatus, Stint
    """
    laps = session.laps.copy()

    # Drop laps with no lap time or clearly erroneous times (pit in/out)
    laps = laps[laps["LapTime"].notna()].copy()
    laps = laps[~laps["PitInTime"].notna()].copy()   # exclude in-laps
    laps = laps[~laps["PitOutTime"].notna()].copy()  # exclude out-laps

    # Convert timedelta columns to float seconds
    time_cols = {
        "LapTime":    "LapTime_s",
        "Sector1Time":"Sector1_s",
        "Sector2Time":"Sector2_s",
        "Sector3Time":"Sector3_s",
    }
    for src, dst in time_cols.items():
        if src in laps.columns:
            laps[dst] = laps[src].dt.total_seconds()

    # Filter out outliers: laps more than 3 std devs from median (SC laps, etc.)
    median_lap = laps["LapTime_s"].median()
    std_lap    = laps["LapTime_s"].std()
    laps = laps[laps["LapTime_s"] < median_lap + 3 * std_lap].copy()

    keep_cols = [
        "Driver", "LapNumber", "LapTime_s",
        "Sector1_s", "Sector2_s", "Sector3_s",
        "Compound", "TyreLife", "IsPersonalBest",
        "TrackStatus", "Stint",
    ]
    available = [c for c in keep_cols if c in laps.columns]
    return laps[available].reset_index(drop=True)


def get_qualifying_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract best qualifying lap per driver.

    Returns DataFrame with one row per driver:
        Driver, BestQualTime_s, Sector1_s, Sector2_s, Sector3_s, Q1_s, Q2_s, Q3_s
    """
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    for col, dst in [("Sector1Time","Sector1_s"),("Sector2Time","Sector2_s"),("Sector3Time","Sector3_s")]:
        if col in laps.columns:
            laps[dst] = laps[col].dt.total_seconds()

    # Best lap per driver
    best = (
        laps.sort_values("LapTime_s")
            .groupby("Driver")
            .first()
            .reset_index()
    )

    keep = ["Driver", "LapTime_s", "Sector1_s", "Sector2_s", "Sector3_s"]
    available = [c for c in keep if c in best.columns]
    result = best[available].rename(columns={"LapTime_s": "BestQualTime_s"})
    return result


# ── Per-driver race aggregates ─────────────────────────────────────────────

def get_driver_race_summary(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Aggregate per-driver stats from a race session.

    Returns one row per driver:
        Driver, AvgLapTime_s, MedianLapTime_s, CleanAirPace_s,
        Sector1_s, Sector2_s, Sector3_s, TotalLaps,
        AvgTyreLife, PrimaryCompound
    """
    laps = get_race_laps(session)
    if laps.empty:
        return pd.DataFrame()

    # Clean air pace: median lap time during laps with TrackStatus == "1" (green flag)
    green_laps = laps[laps["TrackStatus"] == "1"] if "TrackStatus" in laps.columns else laps

    agg = green_laps.groupby("Driver").agg(
        AvgLapTime_s    = ("LapTime_s", "mean"),
        MedianLapTime_s = ("LapTime_s", "median"),
        CleanAirPace_s  = ("LapTime_s", "median"),  # green flag median = clean air proxy
        Sector1_s       = ("Sector1_s", "mean"),
        Sector2_s       = ("Sector2_s", "mean"),
        Sector3_s       = ("Sector3_s", "mean"),
        TotalLaps       = ("LapNumber", "count"),
    ).reset_index()

    if "TyreLife" in laps.columns:
        tyre_agg = laps.groupby("Driver").agg(
            AvgTyreLife     = ("TyreLife", "mean"),
        ).reset_index()
        agg = agg.merge(tyre_agg, on="Driver", how="left")

    if "Compound" in laps.columns:
        primary_compound = (
            laps.groupby("Driver")["Compound"]
                .agg(lambda x: x.value_counts().index[0])
                .reset_index()
                .rename(columns={"Compound": "PrimaryCompound"})
        )
        agg = agg.merge(primary_compound, on="Driver", how="left")

    return agg


# ── Weather extraction ─────────────────────────────────────────────────────

def get_session_weather(session: fastf1.core.Session) -> dict:
    """
    Extract weather summary from a loaded session.

    Returns dict with keys:
        AirTemp_mean, TrackTemp_mean, Humidity_mean,
        WindSpeed_mean, Rainfall_any (bool)
    """
    if session.weather_data is None or session.weather_data.empty:
        logger.warning("No weather data available for this session")
        return {}

    w = session.weather_data
    return {
        "AirTemp_mean":   round(w["AirTemp"].mean(), 1) if "AirTemp" in w.columns else None,
        "TrackTemp_mean": round(w["TrackTemp"].mean(), 1) if "TrackTemp" in w.columns else None,
        "Humidity_mean":  round(w["Humidity"].mean(), 1) if "Humidity" in w.columns else None,
        "WindSpeed_mean": round(w["WindSpeed"].mean(), 1) if "WindSpeed" in w.columns else None,
        "Rainfall_any":   bool(w["Rainfall"].any()) if "Rainfall" in w.columns else False,
    }


# ── Bulk data collector ────────────────────────────────────────────────────

def collect_season_race_data(year: int) -> pd.DataFrame:
    """
    Collect race summaries for every round in a given season.

    Returns a long-format DataFrame with columns:
        Year, Round, Circuit, Driver, [all driver race summary cols], [weather cols]
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    all_rows = []

    for _, event in schedule.iterrows():
        round_num   = event["RoundNumber"]
        circuit     = event["EventName"]

        logger.info(f"  Processing {year} R{round_num}: {circuit}")

        # Race session
        race_session = load_session(year, round_num, "R")
        if race_session is None:
            logger.warning(f"  Skipping {year} R{round_num} — session load failed")
            continue

        driver_summary = get_driver_race_summary(race_session)
        weather        = get_session_weather(race_session)

        if driver_summary.empty:
            continue

        driver_summary["Year"]    = year
        driver_summary["Round"]   = round_num
        driver_summary["Circuit"] = circuit

        for k, v in weather.items():
            driver_summary[k] = v

        all_rows.append(driver_summary)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def collect_season_qualifying_data(year: int) -> pd.DataFrame:
    """
    Collect qualifying data for every round in a given season.

    Returns a long-format DataFrame with columns:
        Year, Round, Circuit, Driver, BestQualTime_s, Sector1_s, Sector2_s, Sector3_s, GridPosition
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    all_rows = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        circuit   = event["EventName"]

        logger.info(f"  Qualifying {year} R{round_num}: {circuit}")

        qual_session = load_session(year, round_num, "Q")
        if qual_session is None:
            continue

        qual_data = get_qualifying_laps(qual_session)
        if qual_data.empty:
            continue

        # Add grid position (rank by qualifying time)
        qual_data = qual_data.sort_values("BestQualTime_s").reset_index(drop=True)
        qual_data["GridPosition"] = qual_data.index + 1

        # Gap to pole
        pole_time = qual_data["BestQualTime_s"].min()
        qual_data["GapToPole_s"] = (qual_data["BestQualTime_s"] - pole_time).round(4)

        qual_data["Year"]    = year
        qual_data["Round"]   = round_num
        qual_data["Circuit"] = circuit

        all_rows.append(qual_data)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Testing FastF1 client — loading 2024 Bahrain race...")
    sess = load_session(2024, 1, "R")
    if sess:
        summary = get_driver_race_summary(sess)
        print(summary[["Driver", "CleanAirPace_s", "Sector1_s", "Sector2_s", "Sector3_s"]].to_string())
        print(f"\nWeather: {get_session_weather(sess)}")