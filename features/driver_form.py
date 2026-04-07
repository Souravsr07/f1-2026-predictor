"""
features/driver_form.py

Driver form features — captures recent performance trajectory for each driver.

The key insight: F1 driver performance is NOT static. A driver on a hot streak
(Norris in mid-2024) is meaningfully different from the same driver in a slump.
We capture this with exponentially decayed rolling windows.

Features produced (one row per driver per race, computed from prior races only):
    - RollingAvgFinish_5          : exp-decayed avg finish position, 5-race window
    - RollingAvgGridPos_5         : exp-decayed avg grid position, 5-race window
    - PositionsGained_avg         : avg positions gained/lost vs grid (rolling 5)
    - WetPerformanceRating        : performance in wet races relative to dry
    - QualiDeltaVsTeammate_s      : avg quali gap to teammate (seconds, positive = slower)
    - DNFRate_rolling             : rolling DNF rate (reliability proxy)
    - FormMomentum                : slope of finish position trend (negative = improving)
    - CircuitTypeFormScore        : form specifically at this circuit type
    - SeasonPointsRatio           : driver's points / championship leader's points
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FORM_WINDOW,
    FORM_DECAY_ALPHA,
    DRIVER_TEAM_2026,
    DRIVER_TEAM_SWITCHES_2026,
    ROOKIES_2026,
    F2_PRIOR_QUALI_DELTA,
)


def _exponential_weights(n: int, alpha: float = FORM_DECAY_ALPHA) -> np.ndarray:
    """
    Exponential decay weights for a rolling window.
    Most recent race has weight 1.0, each prior race multiplied by alpha.
    e.g. alpha=0.7, n=5: [0.24, 0.34, 0.49, 0.70, 1.00]
    """
    weights = np.array([alpha ** (n - 1 - i) for i in range(n)])
    return weights / weights.sum()


def compute_rolling_driver_form(
    results: pd.DataFrame,
    window: int = FORM_WINDOW,
    alpha: float = FORM_DECAY_ALPHA,
) -> pd.DataFrame:
    """
    Compute rolling form features for every driver at every race.

    This is a leakage-safe rolling calculation: features at race R
    are computed using only races 1..R-1 (strictly prior).

    Parameters
    ----------
    results : Master results DataFrame (Year, Round, Driver, FinishPosition,
              GridPosition, DNF, QualPosition, BestQualTime_s, Team, Points)
    window  : Number of prior races to consider
    alpha   : Exponential decay factor (closer to 1 = more uniform)

    Returns
    -------
    DataFrame with original columns + rolling form features
    """
    results = results.sort_values(["Year", "Round"]).copy()

    # Create a global race index (across seasons) for proper ordering
    race_index = (
        results[["Year", "Round"]]
        .drop_duplicates()
        .sort_values(["Year", "Round"])
        .reset_index(drop=True)
    )
    race_index["RaceIdx"] = race_index.index
    results = results.merge(race_index, on=["Year", "Round"], how="left")

    form_rows = []

    for driver in results["Driver"].unique():
        driver_races = results[results["Driver"] == driver].sort_values("RaceIdx").copy()

        for i, (_, current_race) in enumerate(driver_races.iterrows()):
            # Strictly prior races only (no current race data)
            prior = driver_races.iloc[:i]

            row = {
                "Year":   current_race["Year"],
                "Round":  current_race["Round"],
                "Driver": driver,
            }

            if len(prior) == 0:
                # No prior data: use position 10 as neutral prior
                row.update(_neutral_form_prior(driver))
            else:
                recent = prior.tail(window)
                w      = _exponential_weights(len(recent), alpha)

                # Rolling average finish position (weighted)
                if "FinishPosition" in recent.columns:
                    valid_finish = recent["FinishPosition"].dropna()
                    if len(valid_finish) > 0:
                        w_trim = _exponential_weights(len(valid_finish), alpha)
                        row["RollingAvgFinish_5"] = float(np.average(valid_finish, weights=w_trim))
                    else:
                        row["RollingAvgFinish_5"] = 10.0

                # Rolling average grid position
                if "GridPosition" in recent.columns:
                    valid_grid = recent["GridPosition"].replace(0, np.nan).dropna()
                    if len(valid_grid) > 0:
                        w_trim = _exponential_weights(len(valid_grid), alpha)
                        row["RollingAvgGridPos_5"] = float(np.average(valid_grid, weights=w_trim))
                    else:
                        row["RollingAvgGridPos_5"] = 10.0

                # Positions gained vs grid
                if "FinishPosition" in recent.columns and "GridPosition" in recent.columns:
                    gained = (recent["GridPosition"] - recent["FinishPosition"]).dropna()
                    if len(gained) > 0:
                        row["PositionsGained_avg"] = float(gained.mean())
                    else:
                        row["PositionsGained_avg"] = 0.0

                # DNF rate (rolling)
                if "DNF" in recent.columns:
                    row["DNFRate_rolling"] = float(recent["DNF"].mean())
                else:
                    row["DNFRate_rolling"] = 0.08

                # Form momentum: slope of finish position over recent races
                # Negative slope = improving (finishing higher over time)
                if "FinishPosition" in recent.columns and len(recent) >= 3:
                    positions = recent["FinishPosition"].dropna().values
                    if len(positions) >= 2:
                        x = np.arange(len(positions))
                        slope = np.polyfit(x, positions, 1)[0]
                        row["FormMomentum"] = round(float(slope), 4)
                    else:
                        row["FormMomentum"] = 0.0
                else:
                    row["FormMomentum"] = 0.0

                # Season points ratio (driver points / leader points at this moment)
                if "Points" in recent.columns:
                    cumpoints = prior["Points"].sum()
                    # Get max points by any driver at this point
                    all_prior = results[
                        (results["RaceIdx"] < current_race["RaceIdx"]) &
                        (results["Year"] == current_race["Year"])
                    ]
                    if not all_prior.empty:
                        max_pts = all_prior.groupby("Driver")["Points"].sum().max()
                        row["SeasonPointsRatio"] = cumpoints / max_pts if max_pts > 0 else 0.0
                    else:
                        row["SeasonPointsRatio"] = 0.0
                else:
                    row["SeasonPointsRatio"] = 0.0

            form_rows.append(row)

    form_df = pd.DataFrame(form_rows)

    # Merge form features back onto results
    feature_cols = [c for c in form_df.columns if c not in ("Year", "Round", "Driver")]
    merged = results.merge(form_df[["Year", "Round", "Driver"] + feature_cols],
                           on=["Year", "Round", "Driver"], how="left")

    logger.info(f"Rolling form computed: {len(form_df)} driver-race entries")
    return merged


def compute_wet_performance_rating(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each driver's wet-weather performance rating.

    Method: compare average finish position in wet races vs dry races.
    Wet = any race where Rainfall_any=True in the session weather.
    Rating = (avg_dry_finish - avg_wet_finish) / avg_dry_finish
    Positive = performs better in wet relative to their dry baseline.

    Returns original DataFrame + WetPerformanceRating column.
    """
    if "Rainfall_any" not in results.columns:
        logger.warning("No Rainfall_any column — wet performance rating set to 0")
        results["WetPerformanceRating"] = 0.0
        return results

    wet_races = results[results["Rainfall_any"] == True]
    dry_races = results[results["Rainfall_any"] == False]

    wet_avg = wet_races.groupby("Driver")["FinishPosition"].mean().rename("WetAvgFinish")
    dry_avg = dry_races.groupby("Driver")["FinishPosition"].mean().rename("DryAvgFinish")

    rating_df = pd.concat([wet_avg, dry_avg], axis=1)
    rating_df["WetPerformanceRating"] = (
        (rating_df["DryAvgFinish"] - rating_df["WetAvgFinish"]) /
        rating_df["DryAvgFinish"].clip(lower=1)
    ).fillna(0.0)

    results = results.merge(
        rating_df[["WetPerformanceRating"]].reset_index(),
        on="Driver", how="left"
    )
    results["WetPerformanceRating"] = results["WetPerformanceRating"].fillna(0.0)

    logger.info(f"Wet performance ratings computed for {len(rating_df)} drivers")
    return results


def compute_quali_teammate_delta(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each driver's average qualifying gap vs their teammate.

    Method: for each race, find the two drivers in the same team,
    compute delta in BestQualTime_s (positive = slower than teammate).
    Roll this over the season.

    Returns original DataFrame + QualiDeltaVsTeammate_s column.
    """
    if "BestQualTime_s" not in results.columns or "Team" not in results.columns:
        logger.warning("Missing BestQualTime_s or Team — teammate delta set to 0")
        results["QualiDeltaVsTeammate_s"] = 0.0
        return results

    deltas = []
    for (year, round_num, team), group in results.groupby(["Year", "Round", "Team"]):
        group = group.dropna(subset=["BestQualTime_s"])
        if len(group) < 2:
            for _, row in group.iterrows():
                deltas.append({"Year": year, "Round": round_num,
                                "Driver": row["Driver"], "QualiDeltaVsTeammate_s": 0.0})
            continue

        # For each driver, delta = their time - fastest teammate time
        min_time = group["BestQualTime_s"].min()
        for _, row in group.iterrows():
            delta = row["BestQualTime_s"] - min_time
            deltas.append({
                "Year":   year,
                "Round":  round_num,
                "Driver": row["Driver"],
                "QualiDeltaVsTeammate_s": round(float(delta), 4),
            })

    delta_df = pd.DataFrame(deltas)

    # Now roll this: for each race, use the avg delta from the PRIOR window
    results = results.merge(delta_df, on=["Year", "Round", "Driver"], how="left")
    results["QualiDeltaVsTeammate_s"] = results["QualiDeltaVsTeammate_s"].fillna(0.0)

    # Rolling mean per driver (prior races only — shift by 1)
    results = results.sort_values(["Driver", "Year", "Round"])
    results["QualiDeltaVsTeammate_s_rolling"] = (
        results.groupby("Driver")["QualiDeltaVsTeammate_s"]
               .transform(lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
               .fillna(0.0)
    )

    return results


def compute_circuit_type_form(results: pd.DataFrame, circuit_dna: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each driver's average finish position by circuit type
    (street vs permanent vs semi-street).

    Returns original DataFrame + CircuitTypeFormScore column.
    """
    if "Circuit" not in results.columns:
        results["CircuitTypeFormScore"] = 10.0
        return results

    # Map circuit type onto results
    if "CircuitType" in circuit_dna.columns:
        circuit_type_map = circuit_dna["CircuitType"].to_dict()
        results["CircuitType"] = results["Circuit"].map(circuit_type_map)
    else:
        results["CircuitTypeFormScore"] = 10.0
        return results

    type_avg = (
        results.groupby(["Driver", "CircuitType"])["FinishPosition"]
               .mean()
               .reset_index()
               .rename(columns={"FinishPosition": "CircuitTypeFormScore"})
    )

    results = results.merge(type_avg, on=["Driver", "CircuitType"], how="left")
    results["CircuitTypeFormScore"] = results["CircuitTypeFormScore"].fillna(10.0)
    return results


def build_2026_driver_priors(qualifying_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a driver prior table for 2026 — used when we have no 2026 race data yet
    (season start) and to initialise rookie drivers.

    For established drivers: uses their 2025 end-of-season form stats.
    For rookies: uses F2 championship priors from config.
    For team-switchers: applies an adaptation lag discount.

    Returns DataFrame indexed by Driver with columns:
        PriorAvgFinish, PriorWetRating, PriorQualiDelta,
        AdaptationLagFactor, IsRookie, TeamSwitch
    """
    rows = []

    for driver, team in DRIVER_TEAM_2026.items():
        is_rookie     = driver in ROOKIES_2026
        is_switcher   = driver in DRIVER_TEAM_SWITCHES_2026
        races_with_team = DRIVER_TEAM_SWITCHES_2026.get(driver, 99)

        # Adaptation lag: new pairing = 0.85 factor on first 5 races,
        # improving linearly to 1.0 by race 6.
        # Perfect veteran pairing = 1.0.
        if is_rookie:
            adaptation_lag = 0.80
        elif is_switcher and races_with_team < 5:
            adaptation_lag = 0.85 + (0.03 * races_with_team)
        else:
            adaptation_lag = 1.00

        row = {
            "Driver":              driver,
            "Team":                team,
            "IsRookie":            int(is_rookie),
            "IsTeamSwitcher":      int(is_switcher),
            "AdaptationLagFactor": round(adaptation_lag, 3),
        }

        if is_rookie:
            # Use F2 prior — expected quali delta vs midfield
            row["PriorQualiDelta_s"] = F2_PRIOR_QUALI_DELTA.get(driver, 0.0)
            row["PriorAvgFinish"]    = 12.0  # expect midfield at best
            row["PriorWetRating"]    = 0.0   # unknown
        else:
            # Placeholders — will be filled from 2025 rolling form
            # when the pipeline runs; these are cold-start defaults
            row["PriorQualiDelta_s"] = 0.0
            row["PriorAvgFinish"]    = 10.0
            row["PriorWetRating"]    = 0.0

        rows.append(row)

    priors = pd.DataFrame(rows).set_index("Driver")
    logger.info(f"2026 driver priors: {len(priors)} drivers "
                f"({priors['IsRookie'].sum()} rookies, "
                f"{priors['IsTeamSwitcher'].sum()} switchers)")
    return priors


def _neutral_form_prior(driver: str) -> dict:
    """Default form features when a driver has no prior race data."""
    is_rookie = driver in ROOKIES_2026
    return {
        "RollingAvgFinish_5":       12.0 if is_rookie else 10.0,
        "RollingAvgGridPos_5":      11.0 if is_rookie else 10.0,
        "PositionsGained_avg":       0.0,
        "DNFRate_rolling":           0.10 if is_rookie else 0.08,
        "FormMomentum":              0.0,
        "SeasonPointsRatio":         0.0,
    }


if __name__ == "__main__":
    priors = build_2026_driver_priors()
    print(priors[[
        "Team", "IsRookie", "IsTeamSwitcher",
        "AdaptationLagFactor", "PriorAvgFinish"
    ]].to_string())