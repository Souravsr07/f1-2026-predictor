"""
features/circuit_dna.py

Circuit DNA — fingerprints each circuit by its unique performance
characteristics derived from historical lap data.

Why this matters: a driver who is strong at Monza (low-downforce, high-speed)
is not necessarily strong at Monaco (street circuit, mechanical grip). The model
needs to understand circuit archetypes, not just treat every race the same.

Features produced (one row per circuit):
    - SectorBalance_S1/S2/S3     : share of lap time in each sector
    - OvertakeIndex               : historical positions-gained rate (0-1)
    - TyreDegIndex                : avg lap time drop per stint lap (seconds/lap)
    - CircuitType_encoded         : street=2, semi-street=1, permanent=0
    - TrackEvolution              : how much lap time improves over race distance
    - SC_Probability              : historical safety car probability
    - AvgLapTime_s                : baseline lap time reference
    - Sector_dominance            : which sector matters most for lap time (1/2/3)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROCESSED,
    CIRCUITS_2026,
    HIGH_DEG_CIRCUITS,
    LOW_DEG_CIRCUITS,
    SC_PROB,
)


# ── Circuit type encoding ──────────────────────────────────────────────────

CIRCUIT_TYPE_ENCODE = {
    "permanent":   0,
    "semi-street": 1,
    "street":      2,
}

# Historical overtake difficulty (manually curated from 2018–2025 data).
# 0 = nearly impossible (Monaco), 1 = very easy (Monza, Spa, Bahrain)
OVERTAKE_INDEX = {
    "Australia":      0.55,
    "China":          0.60,
    "Japan":          0.45,
    "Bahrain":        0.75,
    "Saudi Arabia":   0.30,
    "Miami":          0.50,
    "Emilia Romagna": 0.40,
    "Monaco":         0.05,
    "Spain":          0.45,
    "Canada":         0.55,
    "Austria":        0.65,
    "Great Britain":  0.55,
    "Belgium":        0.80,
    "Hungary":        0.25,
    "Netherlands":    0.35,
    "Italy":          0.85,
    "Azerbaijan":     0.45,
    "Singapore":      0.20,
    "United States":  0.60,
    "Mexico City":    0.55,
    "São Paulo":      0.60,
    "Las Vegas":      0.55,
    "Qatar":          0.50,
    "Abu Dhabi":      0.45,
}

# Tyre degradation index: how many seconds per lap the tyres drop off.
# Higher = more deg = strategic circuit. Estimated from compound performance data.
TYRE_DEG_INDEX = {
    "Australia":      0.065,
    "China":          0.055,
    "Japan":          0.045,
    "Bahrain":        0.080,   # high deg — abrasive surface
    "Saudi Arabia":   0.040,
    "Miami":          0.058,
    "Emilia Romagna": 0.052,
    "Monaco":         0.020,   # very low deg — cool, short laps
    "Spain":          0.075,   # high deg — hot + long corners
    "Canada":         0.048,
    "Austria":        0.042,
    "Great Britain":  0.060,
    "Belgium":        0.050,
    "Hungary":        0.058,
    "Netherlands":    0.055,
    "Italy":          0.038,   # low deg — mostly straights
    "Azerbaijan":     0.035,
    "Singapore":      0.030,   # low deg — cool night race
    "United States":  0.062,
    "Mexico City":    0.068,   # high altitude = tyre stress
    "São Paulo":      0.055,
    "Las Vegas":      0.040,
    "Qatar":          0.090,   # highest deg on calendar
    "Abu Dhabi":      0.050,
}

# Track evolution factor: how much the track rubbers in during a race.
# High value = big gap between Q3 pace and race pace (affects strategy).
TRACK_EVOLUTION = {
    "Australia":      0.40,
    "China":          0.35,
    "Japan":          0.30,
    "Bahrain":        0.25,
    "Saudi Arabia":   0.55,
    "Miami":          0.45,
    "Emilia Romagna": 0.30,
    "Monaco":         0.60,    # very high — narrow, rubbered-in line only
    "Spain":          0.35,
    "Canada":         0.45,
    "Austria":        0.30,
    "Great Britain":  0.35,
    "Belgium":        0.30,
    "Hungary":        0.40,
    "Netherlands":    0.50,
    "Italy":          0.25,
    "Azerbaijan":     0.60,    # very high — street circuit rubbering
    "Singapore":      0.65,
    "United States":  0.40,
    "Mexico City":    0.35,
    "São Paulo":      0.40,
    "Las Vegas":      0.55,
    "Qatar":          0.30,
    "Abu Dhabi":      0.35,
}


def build_circuit_dna(
    fastf1_laps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the circuit DNA feature table.

    Parameters
    ----------
    fastf1_laps : Optional FastF1 lap data (from data/processed/fastf1_race_laps.parquet).
                  If provided, sector balance is computed from real data.
                  If None, uses hardcoded reference values.

    Returns
    -------
    DataFrame with one row per circuit, indexed by circuit name.
    """
    rows = []

    for circuit_info in CIRCUITS_2026:
        name     = circuit_info["name"]
        ctype    = circuit_info["type"]
        lat      = circuit_info["lat"]
        lon      = circuit_info["lon"]

        row = {
            "Circuit":              name,
            "CircuitType":          ctype,
            "CircuitType_encoded":  CIRCUIT_TYPE_ENCODE.get(ctype, 0),
            "OvertakeIndex":        OVERTAKE_INDEX.get(name, 0.45),
            "TyreDegIndex":         TYRE_DEG_INDEX.get(name, 0.055),
            "TrackEvolution":       TRACK_EVOLUTION.get(name, 0.40),
            "SC_Probability":       SC_PROB.get(ctype, 0.40),
            "IsStreetCircuit":      int(ctype == "street"),
            "Latitude":             lat,
            "Longitude":            lon,
        }

        # Sector balance from real data if available
        if fastf1_laps is not None and not fastf1_laps.empty:
            sb = _compute_sector_balance(fastf1_laps, name)
            row.update(sb)
        else:
            # Reference sector balance (manually set from circuit knowledge)
            row.update(_reference_sector_balance(name))

        # Which sector dominates lap time variance (most predictive of rank)
        s1 = row.get("SectorBalance_S1", 0.33)
        s2 = row.get("SectorBalance_S2", 0.33)
        s3 = row.get("SectorBalance_S3", 0.34)
        row["Sector_dominance"] = int(np.argmax([s1, s2, s3])) + 1

        # High/low deg flag
        row["HighDegCircuit"] = int(name in HIGH_DEG_CIRCUITS)
        row["LowDegCircuit"]  = int(name in LOW_DEG_CIRCUITS)

        rows.append(row)

    dna = pd.DataFrame(rows).set_index("Circuit")
    logger.info(f"Circuit DNA built: {len(dna)} circuits, {len(dna.columns)} features")
    return dna


def _compute_sector_balance(laps: pd.DataFrame, circuit_name: str) -> dict:
    """
    Compute sector time share from real FastF1 lap data.
    Returns fraction of lap time spent in each sector.
    """
    circuit_laps = laps[laps["Circuit"] == circuit_name].copy()

    if circuit_laps.empty or not all(
        c in circuit_laps.columns for c in ["Sector1_s", "Sector2_s", "Sector3_s"]
    ):
        return _reference_sector_balance(circuit_name)

    circuit_laps = circuit_laps.dropna(subset=["Sector1_s", "Sector2_s", "Sector3_s"])
    if circuit_laps.empty:
        return _reference_sector_balance(circuit_name)

    s1_mean = circuit_laps["Sector1_s"].mean()
    s2_mean = circuit_laps["Sector2_s"].mean()
    s3_mean = circuit_laps["Sector3_s"].mean()
    total   = s1_mean + s2_mean + s3_mean

    if total == 0:
        return _reference_sector_balance(circuit_name)

    avg_lap = circuit_laps.groupby("Driver")["LapTime_s"].median().median() \
        if "LapTime_s" in circuit_laps.columns else total

    return {
        "SectorBalance_S1": round(s1_mean / total, 4),
        "SectorBalance_S2": round(s2_mean / total, 4),
        "SectorBalance_S3": round(s3_mean / total, 4),
        "AvgLapTime_s":     round(avg_lap, 3),
    }


def _reference_sector_balance(circuit_name: str) -> dict:
    """
    Hardcoded sector balance reference values (from domain knowledge).
    S1/S2/S3 fractions should sum to ~1.0.
    """
    # (S1_frac, S2_frac, S3_frac, AvgLapTime_s)
    reference = {
        "Australia":      (0.30, 0.38, 0.32, 84.0),
        "China":          (0.29, 0.40, 0.31, 91.5),
        "Japan":          (0.28, 0.42, 0.30, 87.5),
        "Bahrain":        (0.33, 0.35, 0.32, 90.5),
        "Saudi Arabia":   (0.31, 0.37, 0.32, 88.5),
        "Miami":          (0.30, 0.39, 0.31, 86.5),
        "Emilia Romagna": (0.29, 0.40, 0.31, 75.0),
        "Monaco":         (0.30, 0.38, 0.32, 72.5),
        "Spain":          (0.31, 0.37, 0.32, 76.0),
        "Canada":         (0.28, 0.42, 0.30, 70.5),
        "Austria":        (0.27, 0.43, 0.30, 64.5),
        "Great Britain":  (0.30, 0.40, 0.30, 87.0),
        "Belgium":        (0.30, 0.41, 0.29, 105.0),
        "Hungary":        (0.30, 0.39, 0.31, 76.5),
        "Netherlands":    (0.30, 0.40, 0.30, 70.5),
        "Italy":          (0.30, 0.38, 0.32, 81.5),
        "Azerbaijan":     (0.28, 0.41, 0.31, 102.0),
        "Singapore":      (0.31, 0.38, 0.31, 100.0),
        "United States":  (0.30, 0.39, 0.31, 95.0),
        "Mexico City":    (0.30, 0.40, 0.30, 76.0),
        "São Paulo":      (0.28, 0.42, 0.30, 69.5),
        "Las Vegas":      (0.29, 0.41, 0.30, 96.0),
        "Qatar":          (0.30, 0.39, 0.31, 83.5),
        "Abu Dhabi":      (0.30, 0.39, 0.31, 85.0),
    }
    vals = reference.get(circuit_name, (0.33, 0.34, 0.33, 85.0))
    return {
        "SectorBalance_S1": vals[0],
        "SectorBalance_S2": vals[1],
        "SectorBalance_S3": vals[2],
        "AvgLapTime_s":     vals[3],
    }


def get_circuit_features(circuit_name: str, dna: pd.DataFrame) -> pd.Series:
    """
    Retrieve the DNA features for a specific circuit.
    Returns a Series, or raises KeyError if circuit not found.
    """
    if circuit_name not in dna.index:
        raise KeyError(f"Circuit '{circuit_name}' not in DNA table. "
                       f"Available: {list(dna.index)}")
    return dna.loc[circuit_name]


if __name__ == "__main__":
    dna = build_circuit_dna()
    print(dna[[
        "CircuitType", "OvertakeIndex", "TyreDegIndex",
        "SC_Probability", "IsStreetCircuit", "Sector_dominance"
    ]].to_string())