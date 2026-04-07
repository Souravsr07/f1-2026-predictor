"""
features/feature_store.py

Master feature assembler. This is the single entry point for all feature
engineering — it calls every submodule and joins their outputs into one
clean, model-ready DataFrame.

Inputs:
    - Master training data (from data/pipeline.py)
    - Circuit DNA table
    - Constructor scores
    - Driver priors (for 2026)
    - Race-specific data (qualifying + weather)

Output:
    - FEATURE_MATRIX: one row per (driver, race), all features aligned,
      no leakage, properly normalised, saved as parquet.

The final feature set fed to models:
    ┌─────────────────────────────────────────────────────┐
    │ QUALIFYING          GapToPole_s, GridPosition,      │
    │                     QualDeltaFromTeammate_s         │
    │ DRIVER FORM         RollingAvgFinish_5,             │
    │                     FormMomentum, WetRating,        │
    │                     PositionsGained_avg             │
    │ CONSTRUCTOR         DiscountedConstructorScore,     │
    │                     RegDiscountLambda, PU_Tier      │
    │ CIRCUIT             OvertakeIndex, TyreDegIndex,    │
    │                     CircuitType_encoded, SC_Prob    │
    │ RACE CONTEXT        WeatherRiskScore, IsWetRace,    │
    │                     PitStrategyClass, TyreDegFactor │
    │ 2026 SPECIFIC       AdaptationLagFactor, IsRookie   │
    │                     CircuitHistory features         │
    └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, TARGET_YEAR, TRAINING_YEARS

from features.circuit_dna     import build_circuit_dna, get_circuit_features
from features.driver_form     import (
    compute_rolling_driver_form,
    compute_wet_performance_rating,
    compute_quali_teammate_delta,
    compute_circuit_type_form,
    build_2026_driver_priors,
)
from features.constructor     import (
    compute_constructor_scores,
    map_driver_constructor_features,
)
from features.race_features   import (
    compute_qualifying_features,
    compute_race_context_features,
    compute_adaptation_lag_features,
    add_circuit_history_features,
)
from utils.weather_client import get_historical_race_weather


# ── Final feature columns fed to the model ────────────────────────────────
MODEL_FEATURES = [
    # Qualifying
    "GapToPole_s",
    "GridPosition",
    "GridPositionNorm",
    "QualDeltaFromTeammate_s",
    "ExpectedPositionsGained",

    # Driver form (rolling, leakage-safe)
    "RollingAvgFinish_5",
    "RollingAvgGridPos_5",
    "PositionsGained_avg",
    "DNFRate_rolling",
    "FormMomentum",
    "SeasonPointsRatio",
    "WetPerformanceRating",
    "QualiDeltaVsTeammate_s_rolling",
    "CircuitTypeFormScore",

    # Constructor (λ-discounted for 2026)
    "DiscountedConstructorScore",
    "RawConstructorScore",
    "RegDiscountLambda",
    "RegUncertaintySigma",
    "PU_Tier",
    "ResourceTier",
    "ConstructorRiskIndex",

    # Circuit DNA
    "OvertakeIndex",
    "TyreDegIndex",
    "CircuitType_encoded",
    "IsStreetCircuit",
    "SC_Probability",
    "TrackEvolution",
    "SectorBalance_S1",
    "SectorBalance_S2",
    "SectorBalance_S3",
    "Sector_dominance",

    # Race context
    "WeatherRiskScore",
    "RainProbability",
    "Temperature_c",
    "IsWetRace",
    "SC_Probability_Adjusted",
    "PitStrategyClass",
    "TyreDegRaceFactor",

    # Car performance (rolling constructor pace — captures mid-season development)
    "CarMomentum_5race",
    "CarMomentumDelta",
    "TeamQualiGap_s",

    # 2026-specific
    "AdaptationLagFactor",
    "IsRookie",

    # Circuit history (driver-specific)
    "CircuitAvgFinish",
    "CircuitBestFinish",
    "CircuitAppearances",
    "CircuitWinRate",
]

TARGET_COL     = "FinishPosition"
WEIGHT_COL     = "SeasonWeight"
DRIVER_ID_COLS = ["Year", "Round", "Circuit", "Driver", "Team"]


def build_training_feature_matrix(
    master_data:     Optional[pd.DataFrame] = None,
    standings_data:  Optional[pd.DataFrame] = None,
    fastf1_laps:     Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full training feature matrix from historical data.

    If DataFrames are not provided, they are loaded from the parquet store.

    Returns
    -------
    DataFrame with DRIVER_ID_COLS + MODEL_FEATURES + TARGET_COL + WEIGHT_COL.
    Ready to feed into any model in models/.
    """
    logger.info("=== Building training feature matrix ===")

    # ── Load data ──────────────────────────────────────────────────────────
    if master_data is None:
        master_data = _load_parquet("master_training_data.parquet")
    if master_data.empty:
        raise ValueError("No training data found. Run data/pipeline.py --mode full first.")

    if standings_data is None:
        standings_data = _load_parquet("ergast_constructor_standings.parquet")

    if fastf1_laps is None:
        fastf1_laps = _load_parquet("fastf1_race_laps.parquet")

    logger.info(f"Raw master data: {len(master_data)} rows")

    # ── Step 1: Circuit DNA ────────────────────────────────────────────────
    logger.info("Step 1: Building circuit DNA...")
    circuit_dna = build_circuit_dna(
        fastf1_laps=fastf1_laps if not fastf1_laps.empty else None
    )

    # ── Step 2: Constructor scores ─────────────────────────────────────────
    logger.info("Step 2: Computing constructor scores...")
    if not standings_data.empty:
        constructor_scores = compute_constructor_scores(
            standings_data, target_year=2025, reference_year=2025
        )
    else:
        from features.constructor import _equal_constructor_scores
        constructor_scores = _equal_constructor_scores()
        logger.warning("No standings data — using equal constructor scores")

    # ── Step 3: Rolling driver form (leakage-safe) ─────────────────────────
    logger.info("Step 3: Computing rolling driver form...")
    data = compute_rolling_driver_form(master_data)
    data = compute_wet_performance_rating(data)
    data = compute_quali_teammate_delta(data)
    data = compute_circuit_type_form(data, circuit_dna)

    # ── Step 4: Constructor features ───────────────────────────────────────
    logger.info("Step 4: Mapping constructor features...")
    data = map_driver_constructor_features(data, constructor_scores)

    # ── Step 5: Circuit DNA features per row ───────────────────────────────
    logger.info("Step 5: Attaching circuit DNA to each race row...")
    data = _merge_circuit_dna(data, circuit_dna)

    # ── Step 6: Qualifying features ────────────────────────────────────────
    logger.info("Step 6: Computing qualifying features...")
    data = _compute_qualifying_features_per_race(data)

    logger.info("Step 7: Adding historical race-context features...")
    data = _add_historical_race_context_per_race(data, circuit_dna)

    logger.info("Step 8: Adding car momentum features...")
    data = _add_car_momentum(data)

    # ── Step 7: Circuit history ────────────────────────────────────────────
    logger.info("Step 9: Adding circuit history features...")
    data = _add_circuit_history_per_race(data, master_data)

    # ── Step 8: Adaptation lag (0 for historical data — no 2026 switchers) ─
    data["AdaptationLagFactor"] = 1.0   # historical baseline
    data["IsRookie"]            = 0     # filled for 2026 prediction

    # ── Step 9: Finalise ───────────────────────────────────────────────────
    logger.info("Step 11: Finalising feature matrix...")
    data = _finalise_feature_matrix(data)

    logger.info(f"Feature matrix: {len(data)} rows × {len(data.columns)} columns")
    logger.info(f"  Target: {TARGET_COL}")
    logger.info(f"  Features: {len(MODEL_FEATURES)} total")
    logger.info(f"  Missing values: {data[MODEL_FEATURES].isnull().sum().sum()}")

    # Save
    out = DATA_PROCESSED / "feature_matrix.parquet"
    data.to_parquet(out, index=False)
    logger.info(f"Saved feature matrix → {out}")

    return data


def build_prediction_row(
    circuit_name: str,
    qualifying_df: pd.DataFrame,
    weather: dict,
    race_number: int,
    year: int = TARGET_YEAR,
    prior_results: Optional[pd.DataFrame] = None,
    constructor_scores: Optional[pd.DataFrame] = None,
    rolling_form: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a feature row for each driver for an upcoming 2026 race.
    This is called at prediction time (not training time).

    Parameters
    ----------
    circuit_name    : e.g. "Bahrain"
    qualifying_df   : DataFrame with Driver, Team, BestQualTime_s, GridPosition
    weather         : Output of weather_client.get_race_weather_forecast()
    race_number     : Round number in 2026 season
    prior_results   : 2026 results so far (for rolling form). None = season start.
    constructor_scores : Pre-computed constructor scores (loads from parquet if None)
    rolling_form    : Pre-computed rolling form (loads from parquet if None)

    Returns
    -------
    DataFrame with one row per driver, all MODEL_FEATURES populated.
    """
    logger.info(f"Building prediction rows for {circuit_name} ({year} Round {race_number})")

    # Load circuit DNA
    circuit_dna_table = build_circuit_dna()
    try:
        circuit_features = get_circuit_features(circuit_name, circuit_dna_table)
    except KeyError:
        logger.warning(f"Circuit {circuit_name} not in DNA — using defaults")
        circuit_features = None

    if prior_results is None and year >= TARGET_YEAR:
        prior_results = _load_live_results_before_round(year, race_number)
    elif prior_results is not None and not prior_results.empty and "Round" in prior_results.columns:
        prior_results = prior_results[prior_results["Round"] < race_number].copy()

    if constructor_scores is None:
        standings = _load_parquet("ergast_constructor_standings.parquet")
        live_constructor_state = _load_live_constructor_state(year)
        if standings.empty or "Year" not in standings.columns:
            from features.constructor import _equal_constructor_scores
            constructor_scores = _equal_constructor_scores()
        else:
            constructor_scores = compute_constructor_scores(
                standings,
                target_year=year,
                reference_year=min(2025, year - 1),
                live_constructor_state=live_constructor_state,
            )

    # Start with qualifying data
    df = qualifying_df.copy()

    # Add qualifying features
    df = compute_qualifying_features(df, circuit_name)

    # Add race context
    df = compute_race_context_features(df, circuit_name, weather, circuit_features)

    # Add adaptation lag (2026-specific)
    df = compute_adaptation_lag_features(df, race_number)

    # Add constructor features
    df = map_driver_constructor_features(df, constructor_scores)

    # Add circuit DNA features
    if circuit_features is not None:
        for col in circuit_dna_table.columns:
            df[col] = circuit_features.get(col, np.nan)

    if rolling_form is None and year >= TARGET_YEAR and prior_results is not None and not prior_results.empty:
        rolling_form = _build_live_rolling_form(prior_results, circuit_dna_table)

    if rolling_form is not None and not rolling_form.empty:
        form_latest = (
            rolling_form.sort_values(["Driver", "Year", "Round"])
            .groupby("Driver")
            .last()
            .reset_index()
        )
        form_cols   = [c for c in form_latest.columns
                       if c.startswith("Rolling") or c.startswith("Form") or
                          c in ("PositionsGained_avg", "DNFRate_rolling",
                                "SeasonPointsRatio", "WetPerformanceRating",
                                "QualiDeltaVsTeammate_s_rolling", "CircuitTypeFormScore")]
        if form_cols:
            df = df.merge(form_latest[["Driver"] + form_cols], on="Driver", how="left")

    df = _add_prediction_car_features(df, year=year, race_number=race_number, prior_results=prior_results)

    history = _load_historical_circuit_history(year, race_number, prior_results)
    if not history.empty:
        df = add_circuit_history_features(df, circuit_name, history)

    # Add 2026 priors for rookies / team switchers
    priors = build_2026_driver_priors()
    for col in ["AdaptationLagFactor", "IsRookie"]:
        if col in priors.columns:
            prior_map = priors[col].to_dict()
            if col in df.columns:
                df[col] = df["Driver"].map(prior_map).fillna(df[col])
            else:
                df[col] = df["Driver"].map(prior_map)

    # Fill any remaining missing features with sensible defaults
    df = _fill_missing_features(df)

    return df


# ── Helpers ────────────────────────────────────────────────────────────────

def _merge_circuit_dna(data: pd.DataFrame, dna: pd.DataFrame) -> pd.DataFrame:
    """Merge circuit DNA features onto each row by circuit name."""
    dna_reset = dna.reset_index()
    data = data.merge(dna_reset, on="Circuit", how="left", suffixes=("", "_dna"))
    # Drop duplicate columns from merge
    dup_cols = [c for c in data.columns if c.endswith("_dna")]
    data = data.drop(columns=dup_cols)
    return data


def _compute_qualifying_features_per_race(data: pd.DataFrame) -> pd.DataFrame:
    """Apply qualifying feature engineering within each race group."""
    if "BestQualTime_s" not in data.columns:
        return data

    result_chunks = []
    for (year, rnd), group in data.groupby(["Year", "Round"]):
        # Infer circuit name
        circuit = group["Circuit"].iloc[0] if "Circuit" in group.columns else "Unknown"
        chunk   = compute_qualifying_features(group.copy(), circuit)
        result_chunks.append(chunk)

    return pd.concat(result_chunks, ignore_index=True) if result_chunks else data


def _add_historical_race_context_per_race(
    data: pd.DataFrame,
    circuit_dna: pd.DataFrame,
) -> pd.DataFrame:
    """Populate weather and race-context features for historical training rows."""
    if not {"Year", "Round", "Circuit"}.issubset(data.columns):
        return data

    result_chunks = []
    for (year, rnd, circuit), group in data.groupby(["Year", "Round", "Circuit"]):
        weather = get_historical_race_weather(str(circuit), year=int(year), round_number=int(rnd))
        circuit_features = circuit_dna.loc[circuit] if circuit in circuit_dna.index else None
        chunk = compute_race_context_features(
            group.copy(),
            str(circuit),
            weather=weather,
            circuit_dna=circuit_features,
        )
        result_chunks.append(chunk)

    return pd.concat(result_chunks, ignore_index=True) if result_chunks else data


def _add_circuit_history_per_race(
    data: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Add circuit history features for each race row."""
    if "Circuit" not in data.columns:
        for col in ["CircuitAvgFinish","CircuitBestFinish","CircuitAppearances","CircuitWinRate"]:
            data[col] = [10.0, 10, 0, 0.0][["CircuitAvgFinish","CircuitBestFinish","CircuitAppearances","CircuitWinRate"].index(col)]
        return data

    result_chunks = []
    for (year, rnd, circuit), group in data.groupby(["Year", "Round", "Circuit"]):
        # Use history BEFORE this race only (leakage prevention)
        prior_history = history[
            (history["Year"] < year) |
            ((history["Year"] == year) & (history["Round"] < rnd))
        ]
        chunk = add_circuit_history_features(group.copy(), circuit, prior_history)
        result_chunks.append(chunk)

    return pd.concat(result_chunks, ignore_index=True) if result_chunks else data


def _finalise_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Select final columns, fill missing values, enforce dtypes."""
    # Keep ID + feature + target columns
    available_features = [f for f in MODEL_FEATURES if f in data.columns]
    missing_features   = [f for f in MODEL_FEATURES if f not in data.columns]

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features, filling with 0: {missing_features}")
        for f in missing_features:
            data[f] = 0.0

    keep_cols = DRIVER_ID_COLS.copy()
    if TARGET_COL in data.columns:
        keep_cols.append(TARGET_COL)
    if WEIGHT_COL in data.columns:
        keep_cols.append(WEIGHT_COL)
    keep_cols += MODEL_FEATURES

    available_keep = [c for c in keep_cols if c in data.columns]
    result         = data[available_keep].copy()

    # Fill remaining NaN
    result = _fill_missing_features(result)

    # Clip outliers
    if "GapToPole_s" in result.columns:
        result["GapToPole_s"] = result["GapToPole_s"].clip(0, 3.0)
    if "RollingAvgFinish_5" in result.columns:
        result["RollingAvgFinish_5"] = result["RollingAvgFinish_5"].clip(1, 20)

    return result


def _fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN feature values with sensible defaults."""
    defaults = {
        "GapToPole_s":                    1.0,
        "GridPosition":                   10,
        "GridPositionNorm":               0.5,
        "QualDeltaFromTeammate_s":        0.0,
        "ExpectedPositionsGained":        1.5,
        "RollingAvgFinish_5":             10.0,
        "RollingAvgGridPos_5":            10.0,
        "PositionsGained_avg":            0.0,
        "DNFRate_rolling":                0.08,
        "FormMomentum":                   0.0,
        "SeasonPointsRatio":              0.0,
        "WetPerformanceRating":           0.0,
        "QualiDeltaVsTeammate_s_rolling": 0.0,
        "CircuitTypeFormScore":           10.0,
        "DiscountedConstructorScore":     0.40,
        "RawConstructorScore":            0.40,
        "RegDiscountLambda":              0.55,
        "RegUncertaintySigma":            0.22,
        "PU_Tier":                        1,
        "ResourceTier":                   1,
        "ConstructorRiskIndex":           0.15,
        "OvertakeIndex":                  0.45,
        "TyreDegIndex":                   0.055,
        "CircuitType_encoded":            0,
        "IsStreetCircuit":                0,
        "SC_Probability":                 0.40,
        "TrackEvolution":                 0.40,
        "SectorBalance_S1":               0.31,
        "SectorBalance_S2":               0.38,
        "SectorBalance_S3":               0.31,
        "Sector_dominance":               2,
        "WeatherRiskScore":               0.15,
        "RainProbability":                0.15,
        "Temperature_c":                  22.0,
        "IsWetRace":                      0,
        "SC_Probability_Adjusted":        0.42,
        "PitStrategyClass":               1,
        "TyreDegRaceFactor":              0.055,
        "CarMomentum_5race":              8.0,
        "CarMomentumDelta":               0.0,
        "TeamQualiGap_s":                 0.5,
        "AdaptationLagFactor":            1.0,
        "IsRookie":                       0,
        "CircuitAvgFinish":               10.0,
        "CircuitBestFinish":              10,
        "CircuitAppearances":             0,
        "CircuitWinRate":                 0.0,
    }
    for col, val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


def _load_parquet(filename: str) -> pd.DataFrame:
    path = DATA_PROCESSED / filename
    if path.exists():
        return pd.read_parquet(path)
    logger.warning(f"{filename} not found in processed data")
    return pd.DataFrame()


def _load_optional_parquet(filename: str) -> pd.DataFrame:
    path = DATA_PROCESSED / filename
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_live_results_before_round(year: int, race_number: int) -> pd.DataFrame:
    if year != TARGET_YEAR:
        return pd.DataFrame()
    results = _load_optional_parquet("2026_live_results.parquet")
    if results.empty or "Round" not in results.columns:
        return pd.DataFrame()
    return results[results["Round"] < race_number].copy()


def _load_live_qualifying_before_round(year: int, race_number: int) -> pd.DataFrame:
    if year != TARGET_YEAR:
        return pd.DataFrame()
    qualifying = _load_optional_parquet("2026_live_qualifying.parquet")
    if qualifying.empty or "Round" not in qualifying.columns:
        return pd.DataFrame()
    return qualifying[qualifying["Round"] < race_number].copy()


def _load_live_sprint_before_round(year: int, race_number: int) -> pd.DataFrame:
    if year != TARGET_YEAR:
        return pd.DataFrame()
    sprint = _load_optional_parquet("2026_live_sprint.parquet")
    if sprint.empty or "Round" not in sprint.columns:
        return pd.DataFrame()
    return sprint[sprint["Round"] < race_number].copy()


def _load_live_constructor_state(year: int) -> pd.DataFrame:
    if year != TARGET_YEAR:
        return pd.DataFrame()
    state = _load_optional_parquet("2026_live_constructor_state.parquet")
    if state.empty:
        return state
    if "Year" in state.columns:
        state = state[state["Year"] == year].copy()
    return state


def _build_live_rolling_form(
    prior_results: pd.DataFrame,
    circuit_dna_table: pd.DataFrame,
) -> pd.DataFrame:
    if prior_results.empty:
        return pd.DataFrame()

    form_input = prior_results.copy()
    live_qualifying = _load_live_qualifying_before_round(
        TARGET_YEAR,
        int(prior_results["Round"].max()) + 1,
    )
    if not live_qualifying.empty:
        qual_cols = ["Year", "Round", "Driver", "BestQualTime_s"]
        available_cols = [col for col in qual_cols if col in live_qualifying.columns]
        form_input = form_input.merge(
            live_qualifying[available_cols],
            on=[col for col in ["Year", "Round", "Driver"] if col in available_cols],
            how="left",
        )

    form_input = compute_rolling_driver_form(form_input)
    form_input = compute_quali_teammate_delta(form_input)
    form_input = compute_circuit_type_form(form_input, circuit_dna_table)
    if "WetPerformanceRating" not in form_input.columns:
        form_input["WetPerformanceRating"] = 0.0
    return form_input


def _load_historical_circuit_history(
    year: int,
    race_number: int,
    prior_results: Optional[pd.DataFrame],
) -> pd.DataFrame:
    history = _load_optional_parquet("master_training_data.parquet")
    frames = []

    if not history.empty:
        if "Year" in history.columns and "Round" in history.columns:
            history = history[
                (history["Year"] < year)
                | ((history["Year"] == year) & (history["Round"] < race_number))
            ].copy()
        frames.append(history)

    if prior_results is not None and not prior_results.empty:
        frames.append(prior_results.copy())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _add_prediction_car_features(
    df: pd.DataFrame,
    year: int,
    race_number: int,
    prior_results: Optional[pd.DataFrame],
) -> pd.DataFrame:
    result = df.copy()

    result["CarMomentum_5race"] = np.nan
    result["CarMomentumDelta"] = np.nan
    result["TeamQualiGap_s"] = np.nan

    if prior_results is None or prior_results.empty or "Team" not in prior_results.columns:
        return result

    team_race_points = (
        prior_results.groupby(["Round", "Team"], as_index=False)["Points"]
        .sum()
        .rename(columns={"Points": "TeamRacePoints"})
    )

    sprint = _load_live_sprint_before_round(year, race_number)
    if not sprint.empty and "SprintPoints" in sprint.columns:
        sprint_team_points = (
            sprint.groupby(["Round", "Team"], as_index=False)["SprintPoints"]
            .sum()
            .rename(columns={"SprintPoints": "SprintTeamPoints"})
        )
        team_race_points = team_race_points.merge(
            sprint_team_points,
            on=["Round", "Team"],
            how="left",
        )
        team_race_points["SprintTeamPoints"] = team_race_points["SprintTeamPoints"].fillna(0.0)
        team_race_points["TeamRacePoints"] = team_race_points["TeamRacePoints"] + team_race_points["SprintTeamPoints"]
    else:
        team_race_points["SprintTeamPoints"] = 0.0

    team_race_points = team_race_points.sort_values(["Team", "Round"])
    team_race_points["CarMomentum_5race"] = (
        team_race_points.groupby("Team")["TeamRacePoints"]
        .transform(lambda values: values.rolling(5, min_periods=1).mean())
    )
    team_race_points["CarMomentum_10race"] = (
        team_race_points.groupby("Team")["TeamRacePoints"]
        .transform(lambda values: values.rolling(10, min_periods=1).mean())
    )
    team_race_points["CarMomentumDelta"] = (
        team_race_points["CarMomentum_5race"] - team_race_points["CarMomentum_10race"]
    )

    latest_team_momentum = (
        team_race_points.sort_values("Round")
        .groupby("Team")
        .last()[["CarMomentum_5race", "CarMomentumDelta"]]
    )
    result["CarMomentum_5race"] = result["Team"].map(latest_team_momentum["CarMomentum_5race"])
    result["CarMomentumDelta"] = result["Team"].map(latest_team_momentum["CarMomentumDelta"])

    live_qualifying = _load_live_qualifying_before_round(year, race_number)
    if not live_qualifying.empty and {"Round", "Team", "BestQualTime_s"}.issubset(live_qualifying.columns):
        pole_by_round = (
            live_qualifying.groupby("Round", as_index=False)["BestQualTime_s"]
            .min()
            .rename(columns={"BestQualTime_s": "PoleTime_s"})
        )
        team_best = (
            live_qualifying.groupby(["Round", "Team"], as_index=False)["BestQualTime_s"]
            .min()
        )
        team_best = team_best.merge(pole_by_round, on="Round", how="left")
        team_best["TeamQualiGap_s"] = (
            team_best["BestQualTime_s"] - team_best["PoleTime_s"]
        ).clip(lower=0.0, upper=3.0)

        recent_team_gap = (
            team_best.sort_values(["Team", "Round"])
            .groupby("Team")
            .tail(3)
            .groupby("Team")["TeamQualiGap_s"]
            .mean()
        )
        result["TeamQualiGap_s"] = result["Team"].map(recent_team_gap)

    return result


def get_feature_summary(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of feature statistics.
    Useful for EDA notebook and README.
    """
    summary = feature_matrix[MODEL_FEATURES].describe().T
    summary["null_count"] = feature_matrix[MODEL_FEATURES].isnull().sum()
    summary["null_pct"]   = (summary["null_count"] / len(feature_matrix) * 100).round(2)
    return summary



def _add_car_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling constructor performance features.

    Car momentum = rolling avg points per race for the constructor
    over the last 5 races. This captures mid-season car development
    (e.g. Mercedes improving through 2025, Red Bull declining).

    Features added:
      CarMomentum_5race   : constructor avg points per race, 5-race rolling
      CarMomentumDelta    : change vs 10-race rolling (is car improving?)
      TeamQualiGap_s      : team's avg gap to fastest qualifier (car pace proxy)
    """
    if "Team" not in data.columns or "Points" not in data.columns:
        data["CarMomentum_5race"]  = 0.0
        data["CarMomentumDelta"]   = 0.0
        data["TeamQualiGap_s"]     = 0.0
        return data

    data = data.sort_values(["Year", "Round"]).copy()

    # Sum points per team per race first
    team_race_pts = (
        data.groupby(["Year", "Round", "Team"])["Points"]
        .sum().reset_index()
        .rename(columns={"Points": "TeamRacePoints"})
    )

    # Rolling 5-race avg per team (shift 1 to avoid leakage)
    team_race_pts = team_race_pts.sort_values(["Team", "Year", "Round"])
    team_race_pts["CarMomentum_5race"] = (
        team_race_pts.groupby("Team")["TeamRacePoints"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0)
    )
    team_race_pts["CarMomentum_10race"] = (
        team_race_pts.groupby("Team")["TeamRacePoints"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0)
    )
    team_race_pts["CarMomentumDelta"] = (
        team_race_pts["CarMomentum_5race"] - team_race_pts["CarMomentum_10race"]
    ).fillna(0)

    data = data.merge(
        team_race_pts[["Year", "Round", "Team",
                        "CarMomentum_5race", "CarMomentumDelta"]],
        on=["Year", "Round", "Team"], how="left"
    )

    # Team qualifying gap: avg how far behind pole the team's best car is
    if "BestQualTime_s" in data.columns:
        race_pole = (
            data.groupby(["Year", "Round"])["BestQualTime_s"]
            .min().reset_index()
            .rename(columns={"BestQualTime_s": "PoleTime_s"})
        )
        team_best = (
            data.groupby(["Year", "Round", "Team"])["BestQualTime_s"]
            .min().reset_index()
            .rename(columns={"BestQualTime_s": "TeamBestQualTime_s"})
        )
        team_best = team_best.merge(race_pole, on=["Year", "Round"], how="left")
        team_best["TeamQualiGap_s"] = (
            team_best["TeamBestQualTime_s"] - team_best["PoleTime_s"]
        ).fillna(1.0).clip(0, 3.0)
        data = data.merge(
            team_best[["Year", "Round", "Team", "TeamQualiGap_s"]],
            on=["Year", "Round", "Team"],
            how="left",
        )
    else:
        data["TeamQualiGap_s"] = 0.0

    data["CarMomentum_5race"]  = data["CarMomentum_5race"].fillna(0)
    data["CarMomentumDelta"]   = data["CarMomentumDelta"].fillna(0)

    return data


if __name__ == "__main__":
    # Smoke test with dummy qualifying data
    dummy_qual = pd.DataFrame({
        "Driver":         list(["VER","NOR","PIA","LEC","HAM","RUS","SAI","ALO",
                                "STR","GAS","ALB","TSU","OCO","BEA","HUL","BOR",
                                "DOO","HAD","ANT","LAW"]),
        "Team":           ["Red Bull","McLaren","McLaren","Ferrari","Ferrari","Mercedes",
                           "Williams","Aston Martin","Aston Martin","Alpine",
                           "Williams","Racing Bulls","Haas","Haas",
                           "Kick Sauber","Kick Sauber","Alpine","Racing Bulls",
                           "Mercedes","Red Bull"],
        "BestQualTime_s": [90.1,90.3,90.5,90.6,91.0,91.1,91.8,91.9,
                           92.2,92.4,91.7,92.1,92.5,92.8,93.0,93.2,
                           92.6,92.3,91.2,91.5],
        "GridPosition":   list(range(1, 21)),
    })

    weather = {
        "rain_probability": 0.10,
        "temperature_c":    28.0,
        "wind_speed_ms":    4.0,
        "humidity":         45.0,
        "is_wet_race":      False,
        "source":           "test",
    }

    df = build_prediction_row(
        circuit_name="Bahrain",
        qualifying_df=dummy_qual,
        weather=weather,
        race_number=1,
    )

    available = [c for c in MODEL_FEATURES if c in df.columns]
    print(f"\nPrediction rows built: {len(df)} drivers × {len(available)} features")
    print(df[["Driver", "GapToPole_s", "DiscountedConstructorScore",
              "AdaptationLagFactor", "IsRookie", "WeatherRiskScore"]].to_string())
