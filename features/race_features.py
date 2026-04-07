"""
features/race_features.py

Race-specific feature engineering — everything that is race-day specific
and doesn't belong in the rolling driver form or constructor modules.

Features produced:
    - GapToPole_s                 : qualifying gap to pole (clipped at 3.0s)
    - GapToPole_pct               : gap as % of pole time
    - GridPosition                : starting grid position
    - GridPositionNorm            : grid / 20 (normalised)
    - ExpectedPositionsGained     : historical avg positions gained from this grid slot
    - SC_Prob_Adjusted            : safety car prob adjusted for circuit + weather
    - WeatherRiskScore            : 0-1 composite weather risk
    - IsWetRace                   : binary wet race flag
    - PitStrategyClass            : 1-stop=0, 2-stop=1, 3-stop=2 (circuit default)
    - TyreDegRaceFactor           : interaction of circuit deg index × tyre compound
    - DriverTeamAdaptationLag     : penalty for team-switch drivers (races 1–5)
    - QualDeltaFromTeammate_s     : this race's qualifying gap vs teammate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    QUALI_GAP_CLIP,
    SC_PROB,
    DRIVER_TEAM_SWITCHES_2026,
    ROOKIES_2026,
)


# ── Historical positions gained per grid slot ──────────────────────────────
# Computed from 2018–2025 race data. How many positions does a driver in
# grid slot X gain on average during the race?
# P1: tends to lose ~0 (can only go back), P20: gains significantly.
EXPECTED_POSITIONS_GAINED = {
    1:  -0.5,   # pole sitter — slight risk of losing positions
    2:  -0.2,
    3:   0.1,
    4:   0.3,
    5:   0.5,
    6:   0.8,
    7:   1.0,
    8:   1.2,
    9:   1.4,
    10:  1.5,
    11:  1.7,
    12:  1.8,
    13:  1.9,
    14:  2.0,
    15:  2.1,
    16:  2.3,
    17:  2.5,
    18:  2.6,
    19:  2.8,
    20:  3.0,
}

# Default pit strategy by circuit (number of stops)
CIRCUIT_PIT_STRATEGY = {
    "Australia":      2,
    "China":          2,
    "Japan":          2,
    "Bahrain":        2,
    "Saudi Arabia":   2,
    "Miami":          2,
    "Emilia Romagna": 1,
    "Monaco":         1,
    "Spain":          2,
    "Canada":         1,
    "Austria":        2,
    "Great Britain":  2,
    "Belgium":        2,
    "Hungary":        2,
    "Netherlands":    2,
    "Italy":          1,
    "Azerbaijan":     1,
    "Singapore":      1,
    "United States":  2,
    "Mexico City":    2,
    "São Paulo":      2,
    "Las Vegas":      1,
    "Qatar":          3,   # historically high deg → 3-stop common
    "Abu Dhabi":      2,
}


def compute_qualifying_features(
    race_df: pd.DataFrame,
    circuit_name: str,
) -> pd.DataFrame:
    """
    Compute qualifying-derived features for a race.

    Parameters
    ----------
    race_df      : DataFrame with Driver, BestQualTime_s, GridPosition, Team columns
    circuit_name : Name of the circuit

    Returns
    -------
    race_df with qualifying feature columns appended.
    """
    df = race_df.copy()

    if "BestQualTime_s" not in df.columns:
        logger.warning("No BestQualTime_s column — qualifying features set to defaults")
        df["GapToPole_s"]   = 0.0
        df["GapToPole_pct"] = 0.0
        return df

    pole_time = df["BestQualTime_s"].min()

    df["GapToPole_s"] = (df["BestQualTime_s"] - pole_time).clip(0, QUALI_GAP_CLIP).round(4)
    df["GapToPole_pct"] = (df["GapToPole_s"] / pole_time * 100).round(4)

    # Grid position normalised to [0, 1]
    if "GridPosition" in df.columns:
        field_size = max(int(df["GridPosition"].max()), 20)
        df["GridPositionNorm"] = (df["GridPosition"] / float(field_size)).round(4)
        df["ExpectedPositionsGained"] = df["GridPosition"].map(EXPECTED_POSITIONS_GAINED)
        overflow_mask = df["ExpectedPositionsGained"].isna()
        if overflow_mask.any():
            df.loc[overflow_mask, "ExpectedPositionsGained"] = (
                3.0 + 0.2 * (df.loc[overflow_mask, "GridPosition"] - 20)
            )
    
    # Qualifying delta vs teammate
    df = _add_teammate_quali_delta(df)

    return df


def compute_race_context_features(
    race_df: pd.DataFrame,
    circuit_name: str,
    weather: dict,
    circuit_dna: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Add race-context features: weather risk, SC probability, pit strategy.

    Parameters
    ----------
    race_df      : DataFrame with driver rows for this race
    circuit_name : Circuit name
    weather      : Weather dict from weather_client (temperature_c, rain_probability, etc.)
    circuit_dna  : Optional Series of circuit DNA features

    Returns
    -------
    race_df with race-context features appended.
    """
    df = race_df.copy()

    # Weather features
    rain_prob = weather.get("rain_probability", 0.15)
    temp      = weather.get("temperature_c", 22.0)
    wind      = weather.get("wind_speed_ms", 3.0)
    humidity  = weather.get("humidity", 60.0)
    is_wet    = int(weather.get("is_wet_race", rain_prob > 0.60))

    df["RainProbability"]  = round(rain_prob, 3)
    df["Temperature_c"]    = round(temp, 1)
    df["WindSpeed_ms"]     = round(wind, 1)
    df["Humidity"]         = round(humidity, 1)
    df["IsWetRace"]        = is_wet

    # Weather risk score: composite for MC variance scaling
    df["WeatherRiskScore"] = round(
        0.60 * rain_prob +
        0.25 * min(wind / 20.0, 1.0) +
        0.15 * (humidity / 100.0),
        3
    )

    # Safety car probability: base from circuit type, adjusted for wet
    circuit_type = "permanent"
    if circuit_dna is not None and "CircuitType" in circuit_dna.index:
        circuit_type = circuit_dna["CircuitType"]

    base_sc_prob = SC_PROB.get(circuit_type, 0.40)
    # Wet conditions increase SC probability by ~20%
    adjusted_sc_prob = min(base_sc_prob + (0.20 * rain_prob), 0.95)
    df["SC_Probability_Adjusted"] = round(adjusted_sc_prob, 3)

    # Pit strategy class: 1-stop=0, 2-stop=1, 3-stop=2
    default_stops = CIRCUIT_PIT_STRATEGY.get(circuit_name, 2)
    # Wet race → often changes to 2-stop minimum (intermediate + slick transition)
    if is_wet:
        default_stops = max(default_stops, 2)
    df["PitStrategyClass"] = default_stops - 1   # encode as 0/1/2

    # Tyre deg factor: from circuit DNA
    if circuit_dna is not None and "TyreDegIndex" in circuit_dna.index:
        tyre_deg = float(circuit_dna["TyreDegIndex"])
        # High temp = higher deg
        temp_factor = 1.0 + max(0, (temp - 25) / 50)
        df["TyreDegRaceFactor"] = round(tyre_deg * temp_factor, 4)
    else:
        df["TyreDegRaceFactor"] = 0.055

    return df


def compute_adaptation_lag_features(
    race_df: pd.DataFrame,
    race_number_in_season: int,
) -> pd.DataFrame:
    """
    Apply driver-team adaptation lag for 2026 team switchers and rookies.

    For the first N races after a driver joins a new team, their performance
    is discounted by the adaptation lag factor from config.
    This reverts to 1.0 after race 5 with the new team.

    Parameters
    ----------
    race_df                : DataFrame with Driver column
    race_number_in_season  : Current race number (1-based)

    Returns
    -------
    race_df with AdaptationLagFactor column.
    """
    df = race_df.copy()

    def get_lag(driver: str) -> float:
        if driver in ROOKIES_2026:
            # Rookie lag decays over first 8 races
            if race_number_in_season <= 3:
                return 0.80
            elif race_number_in_season <= 6:
                return 0.88
            elif race_number_in_season <= 10:
                return 0.94
            return 1.00

        if driver in DRIVER_TEAM_SWITCHES_2026:
            races_prior = DRIVER_TEAM_SWITCHES_2026[driver]
            effective_races = races_prior + race_number_in_season
            if effective_races <= 3:
                return 0.85
            elif effective_races <= 6:
                return 0.92
            elif effective_races <= 10:
                return 0.97
        return 1.00

    df["AdaptationLagFactor"] = df["Driver"].apply(get_lag)
    return df


def compute_historical_circuit_performance(
    driver: str,
    circuit_name: str,
    results_history: pd.DataFrame,
    min_appearances: int = 2,
) -> dict:
    """
    Look up a driver's historical performance at a specific circuit.

    Returns dict with:
        CircuitAvgFinish, CircuitBestFinish, CircuitAppearances, CircuitWinRate
    """
    driver_at_circuit = results_history[
        (results_history["Driver"] == driver) &
        (results_history["Circuit"].str.contains(circuit_name, case=False, na=False))
    ]

    if len(driver_at_circuit) < min_appearances:
        return {
            "CircuitAvgFinish":   10.0,
            "CircuitBestFinish":  10,
            "CircuitAppearances": len(driver_at_circuit),
            "CircuitWinRate":     0.0,
        }

    return {
        "CircuitAvgFinish":   round(driver_at_circuit["FinishPosition"].mean(), 2),
        "CircuitBestFinish":  int(driver_at_circuit["FinishPosition"].min()),
        "CircuitAppearances": len(driver_at_circuit),
        "CircuitWinRate":     round(
            (driver_at_circuit["FinishPosition"] == 1).mean(), 3
        ),
    }


def add_circuit_history_features(
    race_df: pd.DataFrame,
    circuit_name: str,
    results_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add each driver's historical performance at the current circuit.
    Applied row-by-row for each driver in the race.
    """
    df = race_df.copy()
    history_rows = []

    for _, row in df.iterrows():
        hist = compute_historical_circuit_performance(
            row["Driver"], circuit_name, results_history
        )
        hist["Driver"] = row["Driver"]
        history_rows.append(hist)

    hist_df = pd.DataFrame(history_rows)
    return df.merge(hist_df, on="Driver", how="left")


def _add_teammate_quali_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Add this race's qualifying gap vs teammate."""
    if "Team" not in df.columns or "BestQualTime_s" not in df.columns:
        df["QualDeltaFromTeammate_s"] = 0.0
        return df

    deltas = []
    for team, group in df.groupby("Team"):
        valid = group.dropna(subset=["BestQualTime_s"])
        if len(valid) < 2:
            for idx in group.index:
                deltas.append((idx, 0.0))
            continue
        min_time = valid["BestQualTime_s"].min()
        for idx, row in group.iterrows():
            if pd.isna(row["BestQualTime_s"]):
                deltas.append((idx, 0.0))
            else:
                deltas.append((idx, round(row["BestQualTime_s"] - min_time, 4)))

    delta_series = pd.Series(dict(deltas), name="QualDeltaFromTeammate_s")
    df["QualDeltaFromTeammate_s"] = delta_series
    df["QualDeltaFromTeammate_s"] = df["QualDeltaFromTeammate_s"].fillna(0.0)
    return df


if __name__ == "__main__":
    # Quick smoke test with dummy data
    import pandas as pd

    dummy = pd.DataFrame({
        "Driver":         ["VER", "NOR", "LEC", "HAM", "RUS", "PIA", "SAI", "ALO"],
        "Team":           ["Red Bull","McLaren","Ferrari","Ferrari","Mercedes","McLaren","Williams","Aston Martin"],
        "BestQualTime_s": [70.1, 70.3, 70.5, 70.8, 71.0, 70.4, 71.5, 71.8],
        "GridPosition":   [1, 2, 3, 4, 5, 6, 7, 8],
    })

    dummy = compute_qualifying_features(dummy, "Monaco")
    dummy = compute_race_context_features(
        dummy, "Monaco",
        weather={"rain_probability": 0.20, "temperature_c": 22.0,
                 "wind_speed_ms": 3.0, "humidity": 60.0, "is_wet_race": False}
    )
    dummy = compute_adaptation_lag_features(dummy, race_number_in_season=1)

    print(dummy[[
        "Driver", "GapToPole_s", "GridPositionNorm",
        "WeatherRiskScore", "SC_Probability_Adjusted",
        "PitStrategyClass", "AdaptationLagFactor"
    ]].to_string())
