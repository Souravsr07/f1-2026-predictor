"""
Constructor power features.

This module blends historical reference-year strength with live 2026
constructor evidence so prediction-time features react to the real season
instead of being locked to preseason priors.
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
    EARLY_SEASON_PERFORMANCE_OVERRIDE,
    REG_DISCOUNT_LAMBDA,
    REG_UNCERTAINTY_SIGMA,
    TARGET_YEAR,
)


PU_TIER_2026 = {
    "McLaren": 2,
    "Ferrari": 2,
    "Red Bull": 1,
    "Mercedes": 2,
    "Aston Martin": 1,
    "Alpine": 1,
    "Williams": 2,
    "Racing Bulls": 1,
    "Haas": 1,
    "Audi": 0,
    "Cadillac": 0,
}

RESOURCE_TIER = {
    "McLaren": 2,
    "Ferrari": 2,
    "Red Bull": 2,
    "Mercedes": 2,
    "Aston Martin": 1,
    "Alpine": 1,
    "Williams": 1,
    "Racing Bulls": 1,
    "Haas": 0,
    "Audi": 1,
    "Cadillac": 0,
}


def compute_constructor_scores(
    standings_history: pd.DataFrame,
    target_year: int = 2026,
    reference_year: int = 2025,
    live_constructor_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute constructor performance scores with historical discounting and
    live-season adaptation.

    If a processed live constructor table exists for the target year, its
    points table is blended into the reference-year prior. This makes the
    constructor signal respond quickly once the 2026 season has started.
    """
    ref_standings = standings_history.copy()
    if not ref_standings.empty and "Year" in ref_standings.columns:
        ref_standings = ref_standings[ref_standings["Year"] == reference_year].copy()

    if ref_standings.empty:
        logger.warning(f"No standings data for {reference_year} - using equal constructor priors")
        reference_map = {}
        reference_position_map = {}
        reference_points_map = {}
        mean_score = 0.50
    else:
        ref_standings = ref_standings.groupby("Team", as_index=False).agg(
            Points=("Points", "max"),
            Position=("Position", "min"),
        )
        if target_year >= TARGET_YEAR:
            ref_standings["Team"] = ref_standings["Team"].replace(
                {"Kick Sauber": "Audi", "Sauber": "Audi"}
            )
            ref_standings = ref_standings.groupby("Team", as_index=False).agg(
                Points=("Points", "max"),
                Position=("Position", "min"),
            )
        ref_standings["ReferenceScore"] = _normalize_points(ref_standings["Points"])
        reference_map = ref_standings.set_index("Team")["ReferenceScore"].to_dict()
        reference_position_map = ref_standings.set_index("Team")["Position"].to_dict()
        reference_points_map = ref_standings.set_index("Team")["Points"].to_dict()
        mean_score = float(ref_standings["ReferenceScore"].mean())

    if live_constructor_state is None and target_year >= TARGET_YEAR:
        live_constructor_state = _load_live_constructor_state()

    live_map: dict[str, float] = {}
    live_points_map: dict[str, float] = {}
    completed_rounds = 0
    live_weight = 0.0
    if live_constructor_state is not None and not live_constructor_state.empty:
        live_state = live_constructor_state.copy()
        if "Year" in live_state.columns:
            live_state = live_state[live_state["Year"] == target_year]
        if not live_state.empty:
            live_state = (
                live_state.groupby("Team", as_index=False)
                .agg(
                    Points=("Points", "max"),
                    CompletedRounds=("CompletedRounds", "max"),
                    Position=("Position", "min"),
                )
            )
            live_state["LiveConstructorScore"] = _normalize_points(live_state["Points"])
            live_map = live_state.set_index("Team")["LiveConstructorScore"].to_dict()
            live_points_map = live_state.set_index("Team")["Points"].to_dict()
            completed_rounds = int(live_state["CompletedRounds"].max())
            live_weight = min(0.85, 0.25 + 0.15 * completed_rounds)

    overrides = EARLY_SEASON_PERFORMANCE_OVERRIDE or {}
    teams = sorted(
        set(REG_DISCOUNT_LAMBDA)
        | set(reference_map)
        | set(live_map)
        | set(overrides)
    )

    rows = []
    for team in teams:
        lam = REG_DISCOUNT_LAMBDA.get(team, 0.50)
        sigma = REG_UNCERTAINTY_SIGMA.get(team, 0.25)

        ref_raw = reference_map.get(team, mean_score)
        base_discounted = lam * ref_raw + (1 - lam) * mean_score

        live_score = live_map.get(team)
        combined_raw = ref_raw
        discounted = base_discounted
        if live_score is not None:
            combined_raw = (1 - live_weight) * ref_raw + live_weight * live_score
            discounted = (1 - live_weight) * base_discounted + live_weight * live_score

        if team in overrides:
            discounted = float(np.clip(overrides[team], 0.0, 1.0))

        rows.append(
            {
                "Team": team,
                "RawConstructorScore": round(float(combined_raw), 4),
                "DiscountedConstructorScore": round(float(discounted), 4),
                "LiveConstructorScore": round(float(live_score), 4) if live_score is not None else np.nan,
                "LiveRoundsUsed": completed_rounds if live_score is not None else 0,
                "RegDiscountLambda": lam,
                "RegUncertaintySigma": sigma,
                "PU_Tier": PU_TIER_2026.get(team, 1),
                "ResourceTier": RESOURCE_TIER.get(team, 1),
                "Points_2025": reference_points_map.get(team, np.nan),
                "Position_2025": reference_position_map.get(team, np.nan),
                "PointsLive": live_points_map.get(team, np.nan),
            }
        )

    scores = pd.DataFrame(rows).set_index("Team")
    scores["ConstructorRiskIndex"] = (
        scores["RegUncertaintySigma"]
        * (1 - scores["PU_Tier"] / 2)
        * (1 - scores["ResourceTier"] / 2)
    ).round(4)

    if live_weight > 0:
        logger.info(
            "Constructor scores computed for {} teams using {} live rounds (weight {:.2f})".format(
                len(scores), completed_rounds, live_weight
            )
        )
    else:
        logger.info(f"Constructor scores computed for {len(scores)} teams from {reference_year} priors")
    return scores


def compute_inseason_dev_trajectory(
    standings_history: pd.DataFrame,
    year: int,
    window: int = 5,
) -> pd.DataFrame:
    """Compute rolling points-gain trend for each constructor."""
    season = standings_history[standings_history["Year"] == year].copy()
    if season.empty or "Round" not in season.columns:
        return pd.DataFrame(columns=["Team", "InSeasonDevRate"])

    dev_rows = []
    for team in season["Team"].unique():
        team_data = season[season["Team"] == team].sort_values("Round")
        if len(team_data) < 3:
            dev_rows.append({"Team": team, "InSeasonDevRate": 0.0})
            continue

        increments = np.diff(team_data["Points"].values)
        recent_increments = increments[-window:] if len(increments) else np.array([])
        dev_rate = float(np.mean(recent_increments)) if len(recent_increments) else 0.0
        dev_rows.append({"Team": team, "InSeasonDevRate": round(dev_rate, 3)})

    return pd.DataFrame(dev_rows)


def map_driver_constructor_features(
    race_data: pd.DataFrame,
    constructor_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Map constructor-level features onto a driver-level DataFrame."""
    constructor_reset = constructor_scores.reset_index()

    merge_cols = [
        "Team",
        "RawConstructorScore",
        "DiscountedConstructorScore",
        "RegDiscountLambda",
        "RegUncertaintySigma",
        "PU_Tier",
        "ResourceTier",
        "ConstructorRiskIndex",
    ]
    available = [c for c in merge_cols if c in constructor_reset.columns]

    merged = race_data.merge(constructor_reset[available], on="Team", how="left")

    fill_vals = {
        "RawConstructorScore": 0.30,
        "DiscountedConstructorScore": 0.35,
        "RegDiscountLambda": 0.50,
        "RegUncertaintySigma": 0.25,
        "PU_Tier": 1,
        "ResourceTier": 1,
        "ConstructorRiskIndex": 0.20,
    }
    for col, val in fill_vals.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(val)

    return merged


def _equal_constructor_scores() -> pd.DataFrame:
    """Fallback: equal scores for all known 2026 teams."""
    rows = []
    for team, lam in REG_DISCOUNT_LAMBDA.items():
        rows.append(
            {
                "Team": team,
                "RawConstructorScore": 0.50,
                "DiscountedConstructorScore": 0.50,
                "RegDiscountLambda": lam,
                "RegUncertaintySigma": REG_UNCERTAINTY_SIGMA.get(team, 0.20),
                "PU_Tier": PU_TIER_2026.get(team, 1),
                "ResourceTier": RESOURCE_TIER.get(team, 1),
                "ConstructorRiskIndex": 0.15,
            }
        )
    return pd.DataFrame(rows).set_index("Team")


def _load_live_constructor_state() -> pd.DataFrame:
    path = DATA_PROCESSED / "2026_live_constructor_state.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _normalize_points(points: pd.Series) -> pd.Series:
    if points.empty:
        return pd.Series(dtype=float)
    max_pts = points.max()
    min_pts = points.min()
    if pd.isna(max_pts) or pd.isna(min_pts) or max_pts == min_pts:
        return pd.Series(np.full(len(points), 0.50), index=points.index)
    return ((points - min_pts) / (max_pts - min_pts)).round(4)


def print_lambda_table() -> None:
    """Pretty-print the lambda discount table."""
    print("\n2026 Constructor Regulation Discount (lambda)\n")
    print(f"{'Team':<20} {'lam':>6} {'sig':>6} {'PU':>6} {'Risk':>8}")
    print("-" * 55)

    teams_sorted = sorted(REG_DISCOUNT_LAMBDA.items(), key=lambda item: item[1], reverse=True)
    for team, lam in teams_sorted:
        sigma = REG_UNCERTAINTY_SIGMA.get(team, 0.20)
        pu_tier = PU_TIER_2026.get(team, 1)
        risk = round(sigma * (1 - pu_tier / 2) * (1 - RESOURCE_TIER.get(team, 1) / 2), 3)
        print(f"{team:<20} {lam:>6.2f} {sigma:>6.2f} {pu_tier:>6} {risk:>8.3f}")


if __name__ == "__main__":
    print_lambda_table()
