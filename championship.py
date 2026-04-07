"""
championship.py

Season-long championship forecasting.

Runs 10,000 simulated seasons from the current race onward.
Each simulation samples race outcomes using the same Monte Carlo engine
as single-race predictions, accumulating points and tracking who
clinches WDC/WCC.

Outputs:
  - WDC win probability per driver (updates after each race)
  - WCC win probability per constructor
  - Expected final points per driver/team
  - Title clinch race estimate (P50 and P90)
  - Points trajectory with uncertainty bands
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CIRCUITS_2026, DRIVER_TEAM_2026, DATA_PROCESSED,
    REG_UNCERTAINTY_SIGMA, MONTE_CARLO_N_SIMS, RANDOM_SEED
)
from visualisations import _set_style, _driver_color, _save, DRIVER_TEAM, TEAM_COLORS

# F1 2026 points system
POINTS_SYSTEM = {1:25, 2:18, 3:15, 4:12, 5:10,
                 6:8,  7:6,  8:4,  9:2,  10:1}
FASTEST_LAP_BONUS = 1   # for driver in top 10


class ChampionshipForecaster:
    """
    Simulates the remaining F1 season N times to produce
    championship probability distributions.
    """

    def __init__(self, n_sims: int = MONTE_CARLO_N_SIMS, seed: int = RANDOM_SEED):
        self.n_sims = n_sims
        self.rng    = np.random.default_rng(seed)
        self.results_: Optional[pd.DataFrame] = None
        self.wdc_probs_: Optional[pd.DataFrame] = None
        self.wcc_probs_: Optional[pd.DataFrame] = None

    def forecast(
        self,
        current_standings: pd.DataFrame,
        completed_rounds: int,
        driver_strengths: dict[str, float] = None,
        team_strengths:   dict[str, float] = None,
    ) -> dict:
        """
        Run championship forecast from current standings.

        Parameters
        ----------
        current_standings : DataFrame with Driver, Team, Points columns
        completed_rounds  : Number of races already run
        driver_strengths  : Dict{driver: strength_score} from BT/XGB model
        team_strengths    : Dict{team: discounted_score} from constructor model

        Returns
        -------
        dict with wdc_probs, wcc_probs, expected_points, clinch_rounds
        """
        total_rounds     = len(CIRCUITS_2026)
        remaining_rounds = list(range(completed_rounds + 1, total_rounds + 1))
        n_remaining      = len(remaining_rounds)

        if n_remaining == 0:
            return self._season_complete(current_standings)

        drivers  = current_standings["Driver"].tolist()
        teams    = [DRIVER_TEAM_2026.get(d, "Unknown") for d in drivers]
        n        = len(drivers)

        # Initialise points arrays
        current_pts = {
            row["Driver"]: float(row["Points"])
            for _, row in current_standings.iterrows()
        }

        # Strength scores (fall back to equal if not provided)
        if driver_strengths is None:
            driver_strengths = {d: 0.5 for d in drivers}
        if team_strengths is None:
            team_strengths = {t: 0.5 for t in set(teams)}

        # WDC win counter
        wdc_wins = {d: 0 for d in drivers}
        # WCC win counter
        all_teams = list(set(teams))
        wcc_wins  = {t: 0 for t in all_teams}
        # Points accumulator
        total_pts = {d: [] for d in drivers}
        # Clinch round tracker
        clinch_rounds_wdc = []

        base_strengths = np.array([
            driver_strengths.get(d, 0.5) + 0.3 * team_strengths.get(
                DRIVER_TEAM_2026.get(d, "Unknown"), 0.5
            ) for d in drivers
        ])
        reg_sigmas = np.array([
            REG_UNCERTAINTY_SIGMA.get(DRIVER_TEAM_2026.get(d, "Unknown"), 0.20)
            for d in drivers
        ])

        for _ in range(self.n_sims):
            sim_pts = {d: current_pts.get(d, 0.0) for d in drivers}
            clinched_round = total_rounds  # default: last round

            for rnd_idx, rnd in enumerate(remaining_rounds):
                # Sample performance scores for this race
                noise  = self.rng.normal(0, reg_sigmas)
                sc     = self.rng.random() < 0.45
                if sc:
                    noise *= 0.5   # SC bunches the field

                scores = base_strengths + noise + self.rng.normal(0, 0.2, n)
                order  = np.argsort(-scores)   # descending

                # DNF: some drivers fail to finish
                dnf_probs = np.array([
                    0.07 + reg_sigmas[i] * 0.05 for i in range(n)
                ])
                dnf_mask = self.rng.random(n) < dnf_probs
                # DNF drivers go to back of order
                finishers = [i for i in order if not dnf_mask[i]]
                dnfers    = [i for i in order if dnf_mask[i]]
                final_order = finishers + dnfers

                # Award points
                fastest_lap_driver = finishers[0] if finishers else -1
                for pos, driver_idx in enumerate(final_order, 1):
                    pts = POINTS_SYSTEM.get(pos, 0)
                    drv = drivers[driver_idx]
                    sim_pts[drv] = sim_pts.get(drv, 0) + pts

                # Fastest lap bonus (random top-10 finisher)
                if finishers:
                    fl_idx = self.rng.choice(finishers[:min(10, len(finishers))])
                    fl_drv = drivers[fl_idx]
                    if sim_pts.get(fl_drv, 0) > 0:
                        sim_pts[fl_drv] += FASTEST_LAP_BONUS

                # Check if title clinched mathematically
                pts_sorted   = sorted(sim_pts.values(), reverse=True)
                max_pts      = pts_sorted[0]
                second_pts   = pts_sorted[1] if len(pts_sorted) > 1 else 0
                races_left   = total_rounds - (completed_rounds + rnd_idx + 1)
                max_remaining = sum(POINTS_SYSTEM.get(i, 0) for i in range(1, 2)) * races_left
                if max_pts - second_pts > max_remaining * 25:
                    clinched_round = completed_rounds + rnd_idx + 1
                    break

            # WDC winner
            wdc_winner = max(sim_pts, key=sim_pts.get)
            wdc_wins[wdc_winner] += 1
            clinch_rounds_wdc.append(clinched_round)

            # WCC: sum per team
            team_pts = {}
            for d, pts in sim_pts.items():
                t = DRIVER_TEAM_2026.get(d, "Unknown")
                team_pts[t] = team_pts.get(t, 0) + pts
            wcc_winner = max(team_pts, key=team_pts.get)
            wcc_wins[wcc_winner] += 1

            for d in drivers:
                total_pts[d].append(sim_pts.get(d, 0))

        # Compile WDC probabilities
        wdc_df = pd.DataFrame([
            {
                "Driver":    d,
                "Team":      DRIVER_TEAM_2026.get(d, "Unknown"),
                "WDC_Prob":  wdc_wins[d] / self.n_sims,
                "ExpectedFinalPoints": round(np.mean(total_pts[d]), 1),
                "P10_Points": round(np.percentile(total_pts[d], 10), 1),
                "P90_Points": round(np.percentile(total_pts[d], 90), 1),
                "CurrentPoints": current_pts.get(d, 0),
            }
            for d in drivers
        ]).sort_values("WDC_Prob", ascending=False).reset_index(drop=True)

        # WCC probabilities
        wcc_df = pd.DataFrame([
            {"Team": t, "WCC_Prob": wcc_wins[t] / self.n_sims}
            for t in all_teams
        ]).sort_values("WCC_Prob", ascending=False).reset_index(drop=True)

        # Clinch estimate (P50 = median round)
        clinch_p50 = int(np.percentile(clinch_rounds_wdc, 50))
        clinch_p90 = int(np.percentile(clinch_rounds_wdc, 90))

        self.wdc_probs_ = wdc_df
        self.wcc_probs_ = wcc_df

        return {
            "wdc_probs":       wdc_df,
            "wcc_probs":       wcc_df,
            "clinch_round_p50": clinch_p50,
            "clinch_round_p90": clinch_p90,
            "n_remaining":      n_remaining,
            "completed_rounds": completed_rounds,
        }

    def _season_complete(self, standings: pd.DataFrame) -> dict:
        winner = standings.sort_values("Points", ascending=False).iloc[0]
        wdc_df = standings[["Driver"]].copy()
        wdc_df["WDC_Prob"] = 0.0
        wdc_df.loc[wdc_df["Driver"] == winner["Driver"], "WDC_Prob"] = 1.0
        return {"wdc_probs": wdc_df, "wcc_probs": pd.DataFrame(),
                "clinch_round_p50": 0, "n_remaining": 0}

    # ── Visualisations ─────────────────────────────────────────────────────

    def plot_wdc_probabilities(
        self,
        forecast: dict,
        year: int = 2026,
        save: bool = True,
    ) -> plt.Figure:
        """
        Horizontal bar chart of WDC win probabilities.
        Shows current points and expected final points range.
        """
        _set_style()
        df = forecast["wdc_probs"].head(12)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors  = [_driver_color(d) for d in df["Driver"]]

        bars = ax.barh(
            df["Driver"][::-1],
            (df["WDC_Prob"] * 100)[::-1],
            color=colors[::-1], height=0.65, alpha=0.88,
        )

        # Expected points range annotation
        for bar, (_, row) in zip(bars[::-1], df.iterrows()):
            w = bar.get_width()
            ax.text(
                w + 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{row['CurrentPoints']:.0f}pts now  →  "
                f"{row['P10_Points']:.0f}–{row['P90_Points']:.0f} projected",
                va="center", color="#AAAAAA", fontsize=8,
            )

        rounds_done = forecast["completed_rounds"]
        rounds_left = forecast["n_remaining"]
        clinch_rnd  = forecast.get("clinch_round_p50", "?")

        ax.set_xlabel("WDC win probability (%)")
        ax.set_title(
            f"{year} WDC forecast — after R{rounds_done}, "
            f"{rounds_left} races remaining\n"
            f"Title expected to clinch around R{clinch_rnd}",
            pad=12
        )
        ax.set_xlim(0, df["WDC_Prob"].max() * 100 * 1.45)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()

        if save:
            _save(fig, f"wdc_forecast_{year}_r{rounds_done}.png")
        return fig

    def plot_wcc_probabilities(
        self,
        forecast: dict,
        year: int = 2026,
        save: bool = True,
    ) -> plt.Figure:
        """Constructor championship win probability chart."""
        _set_style()
        df = forecast["wcc_probs"]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = [TEAM_COLORS.get(t, "#888888") for t in df["Team"]]

        ax.barh(
            df["Team"][::-1],
            (df["WCC_Prob"] * 100)[::-1],
            color=colors[::-1], height=0.65, alpha=0.88,
        )

        ax.set_xlabel("WCC win probability (%)")
        ax.set_title(f"{year} WCC forecast", pad=12)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()

        if save:
            _save(fig, f"wcc_forecast_{year}.png")
        return fig

    def plot_points_bands(
        self,
        forecast: dict,
        year: int = 2026,
        top_n: int = 6,
        save: bool = True,
    ) -> plt.Figure:
        """
        Expected final points with P10–P90 uncertainty bands.
        Shows who still has a realistic path to the title.
        """
        _set_style()
        df = forecast["wdc_probs"].head(top_n)

        fig, ax = plt.subplots(figsize=(9, 5))

        y_pos  = range(len(df))
        colors = [_driver_color(d) for d in df["Driver"]]

        ax.barh(
            list(df["Driver"]),
            df["ExpectedFinalPoints"],
            color=colors, height=0.5, alpha=0.5,
        )

        # P10–P90 error bars
        xerr_low  = df["ExpectedFinalPoints"] - df["P10_Points"]
        xerr_high = df["P90_Points"] - df["ExpectedFinalPoints"]
        ax.errorbar(
            df["ExpectedFinalPoints"],
            list(df["Driver"]),
            xerr=[xerr_low, xerr_high],
            fmt="none", color="white", alpha=0.6,
            capsize=5, linewidth=2,
        )

        ax.set_xlabel("Projected final points (P10–P90 range)")
        ax.set_title(f"{year} projected final standings", pad=12)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()

        if save:
            _save(fig, f"points_bands_{year}.png")
        return fig


def update_championship_after_race(
    race_results: pd.DataFrame,
    current_standings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update standings after a race. Adds race points to current standings.

    Parameters
    ----------
    race_results       : DataFrame with Driver, FinishPosition, FastestLap
    current_standings  : DataFrame with Driver, Team, Points

    Returns
    -------
    Updated standings DataFrame.
    """
    new_pts = {}
    for _, row in race_results.iterrows():
        pos = int(row.get("FinishPosition", 20))
        pts = POINTS_SYSTEM.get(pos, 0)
        if row.get("FastestLap", False) and pos <= 10:
            pts += FASTEST_LAP_BONUS
        new_pts[row["Driver"]] = pts

    standings = current_standings.copy()
    standings["RacePoints"] = standings["Driver"].map(new_pts).fillna(0)
    standings["Points"]     = standings["Points"] + standings["RacePoints"]
    standings = standings.drop(columns=["RacePoints"])\
                         .sort_values("Points", ascending=False)\
                         .reset_index(drop=True)
    standings["Position"] = standings.index + 1
    return standings


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    drivers = list(DRIVER_TEAM_2026.keys())
    teams   = [DRIVER_TEAM_2026[d] for d in drivers]

    # Simulate mid-season standings after 8 races
    pts_raw   = np.clip(rng.normal(80, 40, len(drivers)), 0, None)
    standings = pd.DataFrame({
        "Driver": drivers,
        "Team":   teams,
        "Points": np.round(pts_raw).astype(int),
    }).sort_values("Points", ascending=False).reset_index(drop=True)

    print("Current standings (simulated mid-season):")
    print(standings.head(8).to_string(index=False))

    forecaster = ChampionshipForecaster(n_sims=2000)
    result     = forecaster.forecast(standings, completed_rounds=8)

    print("\nWDC Forecast:")
    print(result["wdc_probs"][["Driver","Team","WDC_Prob","ExpectedFinalPoints"]].head(8).to_string(index=False))
    print(f"\nWCC Forecast:")
    print(result["wcc_probs"].head(5).to_string(index=False))
    print(f"\nTitle expected to clinch around: R{result['clinch_round_p50']} "
          f"(P90: R{result['clinch_round_p90']})")

    forecaster.plot_wdc_probabilities(result, save=True)
    forecaster.plot_wcc_probabilities(result, save=True)
    forecaster.plot_points_bands(result, save=True)
    print("\nChampionship plots saved to results/plots/")