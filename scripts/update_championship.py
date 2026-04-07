"""
Generate a championship forecast from ingested 2026 live data.

This script is intentionally self-contained so it can work even while the
main repo is still transitioning from the original 20-driver config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CIRCUITS_2026, DATA_PROCESSED, RANDOM_SEED
from scripts.ingest_live_2026_data import ingest_live_data


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TEAM_COLORS = {
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Haas": "#B6BABD",
    "Red Bull": "#3671C6",
    "Alpine": "#FF87BC",
    "Racing Bulls": "#6692FF",
    "Audi": "#00A19B",
    "Williams": "#64C4FF",
    "Cadillac": "#203A72",
    "Aston Martin": "#229971",
}
POINTS_SYSTEM = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def _ensure_live_data() -> dict[str, pd.DataFrame]:
    required = DATA_PROCESSED / "2026_live_results.parquet"
    if required.exists():
        return {
            "results": pd.read_parquet(DATA_PROCESSED / "2026_live_results.parquet"),
            "sprint": pd.read_parquet(DATA_PROCESSED / "2026_live_sprint.parquet")
            if (DATA_PROCESSED / "2026_live_sprint.parquet").exists()
            else pd.DataFrame(),
            "constructors": pd.read_parquet(DATA_PROCESSED / "2026_live_constructor_state.parquet")
            if (DATA_PROCESSED / "2026_live_constructor_state.parquet").exists()
            else pd.DataFrame(),
        }

    ingested = ingest_live_data()
    return {
        "results": ingested.get("2026_live_results", pd.DataFrame()),
        "sprint": ingested.get("2026_live_sprint", pd.DataFrame()),
        "constructors": ingested.get("2026_live_constructor_state", pd.DataFrame()),
    }


def _build_driver_standings(results: pd.DataFrame, sprint: pd.DataFrame) -> pd.DataFrame:
    race_points = (
        results.groupby(["Driver", "Team"], as_index=False)["Points"]
        .sum()
        .rename(columns={"Points": "RacePoints"})
    ) if not results.empty else pd.DataFrame(columns=["Driver", "Team", "RacePoints"])

    sprint_points = (
        sprint.groupby(["Driver", "Team"], as_index=False)["SprintPoints"]
        .sum()
        .rename(columns={"SprintPoints": "SprintPoints"})
    ) if not sprint.empty else pd.DataFrame(columns=["Driver", "Team", "SprintPoints"])

    standings = race_points.merge(sprint_points, on=["Driver", "Team"], how="outer").fillna(0)
    standings["Points"] = standings["RacePoints"] + standings["SprintPoints"]
    standings = standings.sort_values(["Points", "Driver"], ascending=[False, True]).reset_index(drop=True)
    standings["Position"] = standings.index + 1
    return standings[["Position", "Driver", "Team", "Points", "RacePoints", "SprintPoints"]]


def _normalized_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    lo = values.min()
    hi = values.max()
    if hi <= lo:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - lo) / (hi - lo)


def _load_strengths(driver_standings: pd.DataFrame, constructors: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    driver_points_norm = _normalized_series(driver_standings["Points"])
    driver_strengths = pd.Series(driver_points_norm.values, index=driver_standings["Driver"]).to_dict()

    model_path = DATA_PROCESSED / "ensemble_model.pkl"
    if model_path.exists():
        try:
            from models.ensemble import EnsemblePredictor

            ensemble = EnsemblePredictor.load(str(model_path))
            bt_strengths = pd.Series(ensemble.bt.strengths_, dtype=float)
            if not bt_strengths.empty:
                bt_norm = _normalized_series(bt_strengths.reindex(driver_standings["Driver"]).fillna(bt_strengths.mean()))
                for driver in driver_standings["Driver"]:
                    driver_strengths[driver] = float(0.6 * bt_norm.loc[driver] + 0.4 * driver_strengths.get(driver, 0.5))
        except Exception:
            pass

    if constructors.empty:
        team_points = driver_standings.groupby("Team")["Points"].sum().reset_index()
        constructors = team_points.rename(columns={"Points": "Points"})
    team_strengths = pd.Series(
        _normalized_series(constructors["Points"]).values,
        index=constructors["Team"],
    ).to_dict()
    return driver_strengths, team_strengths


def _simulate_season(
    driver_standings: pd.DataFrame,
    constructors: pd.DataFrame,
    n_sims: int = 5000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    total_rounds = len(CIRCUITS_2026)
    completed_rounds = min(driver_standings.get("CompletedRounds", pd.Series(dtype=int)).max() if "CompletedRounds" in driver_standings else 0, total_rounds)
    if completed_rounds == 0:
        completed_rounds = 0

    driver_strengths, team_strengths = _load_strengths(driver_standings, constructors)

    current_points = pd.Series(driver_standings["Points"].values, index=driver_standings["Driver"]).to_dict()
    team_lookup = pd.Series(driver_standings["Team"].values, index=driver_standings["Driver"]).to_dict()
    drivers = driver_standings["Driver"].tolist()

    wdc_wins = {driver: 0 for driver in drivers}
    wcc_wins = {team: 0 for team in sorted(driver_standings["Team"].unique())}
    expected_points = {driver: [] for driver in drivers}

    remaining_rounds = list(range(completed_rounds + 1, total_rounds + 1))
    for _ in range(n_sims):
        sim_points = current_points.copy()
        for _round in remaining_rounds:
            scores = []
            for driver in drivers:
                team = team_lookup[driver]
                base = 0.65 * driver_strengths.get(driver, 0.5) + 0.35 * team_strengths.get(team, 0.5)
                noise = rng.normal(0, 0.12 + 0.08 * (1 - team_strengths.get(team, 0.5)))
                scores.append(base + noise)

            order = np.argsort(-np.array(scores))
            dnf_mask = rng.random(len(drivers)) < 0.05
            finishers = [drivers[idx] for idx in order if not dnf_mask[idx]]
            dnfers = [drivers[idx] for idx in order if dnf_mask[idx]]
            final_order = finishers + dnfers

            for pos, driver in enumerate(final_order, start=1):
                sim_points[driver] = sim_points.get(driver, 0.0) + POINTS_SYSTEM.get(pos, 0)

        winner = max(sim_points, key=sim_points.get)
        wdc_wins[winner] += 1

        team_points = {}
        for driver, points in sim_points.items():
            team = team_lookup[driver]
            team_points[team] = team_points.get(team, 0.0) + points
        wcc_winner = max(team_points, key=team_points.get)
        wcc_wins[wcc_winner] += 1

        for driver in drivers:
            expected_points[driver].append(sim_points[driver])

    wdc = pd.DataFrame(
        [
            {
                "Driver": driver,
                "Team": team_lookup[driver],
                "CurrentPoints": current_points[driver],
                "WDC_Prob": wdc_wins[driver] / n_sims,
                "ExpectedFinalPoints": round(float(np.mean(expected_points[driver])), 1),
                "P10_Points": round(float(np.percentile(expected_points[driver], 10)), 1),
                "P90_Points": round(float(np.percentile(expected_points[driver], 90)), 1),
            }
            for driver in drivers
        ]
    ).sort_values(["WDC_Prob", "ExpectedFinalPoints"], ascending=[False, False]).reset_index(drop=True)

    wcc = pd.DataFrame(
        [{"Team": team, "WCC_Prob": prob / n_sims} for team, prob in wcc_wins.items()]
    ).sort_values("WCC_Prob", ascending=False).reset_index(drop=True)

    return wdc, wcc


def _plot_barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [TEAM_COLORS.get(team, "#888888") for team in df.get("Team", pd.Series([""] * len(df)))]
    ax.barh(df[label_col][::-1], (df[value_col] * 100)[::-1], color=colors[::-1], alpha=0.88)
    ax.set_xlabel("Probability (%)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    live = _ensure_live_data()
    results = live["results"]
    sprint = live["sprint"]
    constructors = live["constructors"]

    if results.empty:
        print("No live 2026 results available. Run scripts/ingest_live_2026_data.py first.")
        return

    driver_standings = _build_driver_standings(results, sprint)
    completed_rounds = int(results["Round"].nunique())
    driver_standings["CompletedRounds"] = completed_rounds

    if constructors.empty:
        constructors = (
            driver_standings.groupby("Team", as_index=False)["Points"].sum()
            .sort_values("Points", ascending=False)
            .reset_index(drop=True)
        )
        constructors["Position"] = constructors.index + 1
        constructors["Year"] = 2026
        constructors["CompletedRounds"] = completed_rounds

    wdc, wcc = _simulate_season(driver_standings, constructors)
    wdc.to_csv(RESULTS_DIR / "wdc_forecast_2026.csv", index=False)
    wcc.to_csv(RESULTS_DIR / "wcc_forecast_2026.csv", index=False)
    driver_standings.to_csv(RESULTS_DIR / "driver_standings_2026.csv", index=False)
    constructors.to_csv(RESULTS_DIR / "constructor_standings_2026.csv", index=False)

    _plot_barh(
        wdc.head(12),
        "Driver",
        "WDC_Prob",
        f"2026 WDC forecast after R{completed_rounds}",
        f"wdc_forecast_2026_r{completed_rounds}.png",
    )
    _plot_barh(
        wcc.head(11),
        "Team",
        "WCC_Prob",
        f"2026 WCC forecast after R{completed_rounds}",
        f"wcc_forecast_2026_r{completed_rounds}.png",
    )

    print(f"Saved WDC/WCC forecasts after R{completed_rounds}")


if __name__ == "__main__":
    main()
