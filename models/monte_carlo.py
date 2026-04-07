"""
models/monte_carlo.py

Monte Carlo race simulation engine.

This is the probabilistic heart of the predictor. Instead of outputting a
single predicted finishing order, we simulate N=10,000 races, each with
stochastic events injected:
  - Safety car deployment (probability from circuit DNA + weather)
  - Driver DNFs (per-constructor reliability + random failure)
  - Weather changes mid-race (rain probability)
  - Pit stop variance (timing and tyre choice uncertainty)
  - Overtaking success rates (circuit-specific)

The output of 10,000 simulations is a full probability distribution:
  P(driver i finishes in position k) for all i, k

This is far more informative than a point estimate. A result like:
  VER: P(win)=38%, P(podium)=72%, P(top10)=91%
  NOR: P(win)=31%, P(podium)=65%, P(top10)=89%

...is honest about uncertainty in a way that "VER wins" is not.

Critically for 2026: we widen the variance for all constructors by their
RegUncertaintySigma, encoding our genuine uncertainty about car performance.
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
    MONTE_CARLO_N_SIMS,
    RANDOM_SEED,
    DNF_PROB_BASELINE,
    SC_PROB,
    REG_UNCERTAINTY_SIGMA,
    DRIVER_TEAM_2026,
)


# Team tier for DNF probability lookup
TEAM_TIER = {
    "McLaren":       "top",
    "Ferrari":       "top",
    "Red Bull":      "top",
    "Mercedes":      "top",
    "Aston Martin":  "mid",
    "Alpine":        "mid",
    "Williams":      "mid",
    "Racing Bulls":  "mid",
    "Haas":          "lower",
    "Kick Sauber":   "lower",
    "Audi":          "lower",
    "Cadillac":      "lower",
}


class MonteCarloSimulator:
    """
    Monte Carlo race simulator.

    Input : per-driver feature rows (output of feature_store.build_prediction_row())
            + pre-computed rank scores from XGBoost and/or Bradley-Terry
    Output: probability distribution over finishing positions
    """

    def __init__(
        self,
        n_sims: int = MONTE_CARLO_N_SIMS,
        seed: int = RANDOM_SEED,
    ):
        self.n_sims = n_sims
        self.rng    = np.random.default_rng(seed)
        self.last_simulation_results_ = None

    def simulate(
        self,
        race_df: pd.DataFrame,
        xgb_scores: np.ndarray = None,
        bt_strengths: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Run N Monte Carlo race simulations.

        Parameters
        ----------
        race_df      : Feature DataFrame (one row per driver, 20 rows for a full grid)
        xgb_scores   : Raw XGBoost rank scores (optional, improves base performance)
        bt_strengths : Bradley-Terry β values (optional)

        Returns
        -------
        DataFrame with columns:
            Driver, Team,
            WinProb, P2Prob, P3Prob, PodiumProb, Top5Prob, Top10Prob,
            DNFProb, ExpectedPos, PosStdDev,
            P1_through_P20 (full position distribution)
        """
        drivers = race_df["Driver"].tolist()
        teams   = race_df["Team"].tolist() if "Team" in race_df.columns \
            else [DRIVER_TEAM_2026.get(d, "Unknown") for d in drivers]
        n       = len(drivers)

        logger.info(f"Running {self.n_sims:,} simulations for {n} drivers...")

        # ── Build base performance scores ──────────────────────────────────
        base_scores = self._build_base_scores(
            race_df, xgb_scores, bt_strengths, drivers
        )

        # ── Extract simulation parameters ──────────────────────────────────
        dnf_probs      = self._get_dnf_probabilities(race_df, teams)
        sc_prob        = self._get_sc_probability(race_df)
        weather_risk   = float(race_df["WeatherRiskScore"].mean()) \
            if "WeatherRiskScore" in race_df.columns else 0.15
        reg_sigmas     = self._get_reg_sigmas(teams)

        # Position count matrix: sim_positions[driver_idx, position_idx] = count
        position_counts = np.zeros((n, n), dtype=np.int32)
        dnf_counts      = np.zeros(n, dtype=np.int32)

        # ── Main simulation loop ───────────────────────────────────────────
        for sim in range(self.n_sims):
            finishing_order = self._simulate_one_race(
                base_scores    = base_scores,
                reg_sigmas     = reg_sigmas,
                dnf_probs      = dnf_probs,
                sc_prob        = sc_prob,
                weather_risk   = weather_risk,
                n_drivers      = n,
            )

            # finishing_order[k] = driver index who finished in position k+1
            # DNF drivers get position n-dnf_count .. n
            for pos, driver_idx in enumerate(finishing_order):
                if driver_idx >= 0:
                    position_counts[driver_idx, pos] += 1
                else:
                    dnf_counts[abs(driver_idx) - 1] += 1

        # ── Compile results ────────────────────────────────────────────────
        results = self._compile_results(
            drivers, teams, position_counts, dnf_counts
        )

        self.last_simulation_results_ = results
        logger.info(f"Simulation complete. Predicted winner: "
                    f"{results.iloc[0]['Driver']} "
                    f"(P(win)={results.iloc[0]['WinProb']:.1%})")
        return results

    # ── Single race simulation ─────────────────────────────────────────────

    def _simulate_one_race(
        self,
        base_scores:  np.ndarray,
        reg_sigmas:   np.ndarray,
        dnf_probs:    np.ndarray,
        sc_prob:      float,
        weather_risk: float,
        n_drivers:    int,
    ) -> list[int]:
        """
        Simulate one race. Returns list of driver indices in finishing order.
        DNF drivers are represented as negative indices.
        """
        # ── Step 1: Sample performance scores with noise ───────────────────
        # Each driver's race performance = base_score + constructor_noise + random
        # constructor_noise is drawn from N(0, σ_reg) — encodes 2026 uncertainty
        constructor_noise = self.rng.normal(0, reg_sigmas)
        random_noise      = self.rng.normal(0, 0.3, n_drivers)
        scores            = base_scores + constructor_noise + random_noise

        # ── Step 2: Safety car event ───────────────────────────────────────
        # SC bunches the field — compress score differences by 40%
        if self.rng.random() < sc_prob:
            sc_timing = self.rng.random()   # SC comes early/late
            # Later SC compresses field more (drivers have spread out)
            compression = 0.4 + 0.3 * sc_timing
            scores      = scores * (1 - compression) + scores.mean() * compression

        # ── Step 3: Weather event ──────────────────────────────────────────
        # Random rain onset mid-race → shuffle performance scores partially
        if self.rng.random() < weather_risk * 0.3:
            wet_shuffle = self.rng.normal(0, 0.5, n_drivers)
            scores      = scores + wet_shuffle

        # ── Step 4: DNF events ────────────────────────────────────────────
        # Each driver independently fails with their DNF probability
        dnf_mask = self.rng.random(n_drivers) < dnf_probs

        # ── Step 5: Build finishing order ──────────────────────────────────
        finishers    = [(i, scores[i]) for i in range(n_drivers) if not dnf_mask[i]]
        dnf_drivers  = [i for i in range(n_drivers) if dnf_mask[i]]

        # Sort finishers by score (descending = better performance)
        finishers.sort(key=lambda x: -x[1])

        finishing_order = [idx for idx, _ in finishers]

        # DNF drivers get random positions at the back
        if dnf_drivers:
            self.rng.shuffle(dnf_drivers)
            finishing_order.extend([-(driver_idx + 1) for driver_idx in dnf_drivers])

        return finishing_order

    # ── Score construction ─────────────────────────────────────────────────

    def _build_base_scores(
        self,
        race_df:     pd.DataFrame,
        xgb_scores:  Optional[np.ndarray],
        bt_strengths: Optional[np.ndarray],
        drivers:     list[str],
    ) -> np.ndarray:
        """
        Build base performance scores for each driver.
        Combines XGBoost, Bradley-Terry, and feature-derived signals.
        """
        n = len(drivers)
        scores = np.zeros(n)

        # XGBoost contribution (normalised to [0,1] range)
        if xgb_scores is not None and len(xgb_scores) == n:
            xgb_norm  = (xgb_scores - xgb_scores.min()) / \
                        (xgb_scores.max() - xgb_scores.min() + 1e-8)
            scores   += 0.5 * xgb_norm

        # Bradley-Terry contribution
        if bt_strengths is not None and len(bt_strengths) == n:
            bt_norm   = (bt_strengths - bt_strengths.min()) / \
                        (bt_strengths.max() - bt_strengths.min() + 1e-8)
            scores   += 0.3 * bt_norm

        # Feature-derived signals (when model scores not available)
        if xgb_scores is None and bt_strengths is None:
            scores = self._scores_from_features(race_df)
            return scores

        # Add qualifying signal (always useful)
        if "GapToPole_s" in race_df.columns:
            gap      = race_df["GapToPole_s"].fillna(1.5).values
            qual_sig = 1.0 - (gap / gap.max().clip(1))
            scores  += 0.2 * qual_sig

        return scores

    def _scores_from_features(self, race_df: pd.DataFrame) -> np.ndarray:
        """
        Derive performance scores directly from features.
        Used when XGBoost/BT scores are not available (e.g. cold start).
        """
        n      = len(race_df)
        scores = np.zeros(n)

        # Qualifying time (most predictive single feature)
        if "GapToPole_s" in race_df.columns:
            gap     = race_df["GapToPole_s"].fillna(1.5).values
            scores += 1.0 - (gap / (gap.max() + 1e-8))

        # Constructor power (λ-discounted)
        if "DiscountedConstructorScore" in race_df.columns:
            cs      = race_df["DiscountedConstructorScore"].fillna(0.4).values
            scores += 0.5 * cs

        # Rolling form
        if "RollingAvgFinish_5" in race_df.columns:
            form    = race_df["RollingAvgFinish_5"].fillna(10).values
            form_sc = 1.0 - ((form - 1) / 19.0)
            scores += 0.3 * form_sc

        # Adaptation lag penalty
        if "AdaptationLagFactor" in race_df.columns:
            lag     = race_df["AdaptationLagFactor"].fillna(1.0).values
            scores *= lag

        return scores

    # ── Parameter helpers ──────────────────────────────────────────────────

    def _get_dnf_probabilities(
        self, race_df: pd.DataFrame, teams: list[str]
    ) -> np.ndarray:
        """
        Get per-driver DNF probability.
        For 2026: increase baseline by RegUncertaintySigma (new cars fail more).
        """
        dnf_probs = np.array([
            DNF_PROB_BASELINE.get(TEAM_TIER.get(t, "mid"), 0.10)
            for t in teams
        ])

        # 2026 reg-reset adds extra mechanical unreliability
        for i, team in enumerate(teams):
            sigma        = REG_UNCERTAINTY_SIGMA.get(team, 0.20)
            dnf_probs[i] += sigma * 0.05   # e.g. Kick Sauber σ=0.30 → +1.5% DNF

        # From rolling DNF rate if available
        if "DNFRate_rolling" in race_df.columns:
            historical_dnf = race_df["DNFRate_rolling"].fillna(0.08).clip(0.02, 0.25).values
            # Early-season rolling DNF rates are extremely noisy, so only let
            # them nudge the baseline instead of dominate it.
            dnf_probs      = 0.8 * dnf_probs + 0.2 * historical_dnf

        return dnf_probs.clip(0.02, 0.20)

    def _get_sc_probability(self, race_df: pd.DataFrame) -> float:
        """Get safety car probability for this race."""
        if "SC_Probability_Adjusted" in race_df.columns:
            return float(race_df["SC_Probability_Adjusted"].iloc[0])
        if "SC_Probability" in race_df.columns:
            return float(race_df["SC_Probability"].iloc[0])
        return 0.40

    def _get_reg_sigmas(self, teams: list[str]) -> np.ndarray:
        """Get per-driver constructor uncertainty sigma (for score sampling)."""
        return np.array([
            REG_UNCERTAINTY_SIGMA.get(t, 0.20) for t in teams
        ])

    # ── Results compilation ────────────────────────────────────────────────

    def _compile_results(
        self,
        drivers:         list[str],
        teams:           list[str],
        position_counts: np.ndarray,
        dnf_counts:      np.ndarray,
    ) -> pd.DataFrame:
        """Compile position count matrix into probability DataFrame."""
        n    = len(drivers)
        rows = []

        for i, driver in enumerate(drivers):
            total_dnfs     = dnf_counts[i]

            pos_probs = position_counts[i] / (self.n_sims)   # P(pos k)

            # Expected finishing position (including DNF as one position behind
            # the classified runners) and its full variance.
            dnf_prob = total_dnfs / self.n_sims
            all_probs = np.append(pos_probs, dnf_prob)
            all_positions = np.arange(1, n + 2)
            exp_pos = np.sum(all_positions * all_probs)
            pos_std = np.sqrt(np.sum(((all_positions - exp_pos) ** 2) * all_probs))

            row = {
                "Driver":    driver,
                "Team":      teams[i],
                "WinProb":   round(float(pos_probs[0]), 4),
                "P2Prob":    round(float(pos_probs[1]) if n > 1 else 0, 4),
                "P3Prob":    round(float(pos_probs[2]) if n > 2 else 0, 4),
                "PodiumProb":round(float(pos_probs[:3].sum()), 4),
                "Top5Prob":  round(float(pos_probs[:5].sum()), 4),
                "Top10Prob": round(float(pos_probs[:10].sum()), 4),
                "DNFProb":   round(float(dnf_prob), 4),
                "ExpectedPos":round(float(exp_pos), 2),
                "PosStdDev": round(float(pos_std), 2),
            }

            # Full position distribution P1..P20
            for k in range(n):
                row[f"P{k+1}_prob"] = round(float(pos_probs[k]), 4)

            rows.append(row)

        result = pd.DataFrame(rows)
        result = result.sort_values("WinProb", ascending=False).reset_index(drop=True)
        result["PredictedPos"] = result.index + 1
        return result

    def get_podium_matrix(self) -> pd.DataFrame:
        """
        Returns a clean podium probability matrix (Driver × P1/P2/P3).
        Perfect for the LinkedIn post heatmap visualisation.
        """
        if self.last_simulation_results_ is None:
            raise RuntimeError("No simulation run yet. Call simulate() first.")

        res = self.last_simulation_results_
        return res[["Driver", "Team", "WinProb", "P2Prob", "P3Prob", "PodiumProb"]]\
               .sort_values("WinProb", ascending=False)\
               .reset_index(drop=True)

    def get_position_distribution(self, driver: str) -> pd.Series:
        """
        Return the full position probability distribution for a single driver.
        """
        if self.last_simulation_results_ is None:
            raise RuntimeError("No simulation results. Call simulate() first.")

        row = self.last_simulation_results_[
            self.last_simulation_results_["Driver"] == driver
        ]
        if row.empty:
            raise ValueError(f"Driver {driver} not found in results")

        prob_cols = [c for c in row.columns if c.startswith("P") and c.endswith("_prob")]
        series    = row[prob_cols].iloc[0]
        series.index = [int(c.split("_")[0][1:]) for c in series.index]
        series.name  = driver
        return series.sort_index()


if __name__ == "__main__":
    import pandas as pd

    # Smoke test with dummy data
    dummy = pd.DataFrame({
        "Driver":                      ["VER","NOR","LEC","HAM","RUS","PIA","SAI","ALO",
                                        "STR","GAS","ALB","TSU","OCO","BEA","HUL","BOR",
                                        "DOO","HAD","ANT","LAW"],
        "Team":                        ["Red Bull","McLaren","Ferrari","Ferrari","Mercedes",
                                        "McLaren","Williams","Aston Martin","Aston Martin","Alpine",
                                        "Williams","Racing Bulls","Haas","Haas","Kick Sauber",
                                        "Kick Sauber","Alpine","Racing Bulls","Mercedes","Red Bull"],
        "GapToPole_s":                 [0.0,0.2,0.5,0.9,1.1,0.4,1.7,1.8,2.2,2.4,
                                        1.6,2.0,2.5,2.7,2.9,3.0,2.6,2.2,1.1,1.4],
        "DiscountedConstructorScore":  [0.56,0.68,0.65,0.65,0.62,0.68,0.48,0.43,0.43,0.42,
                                        0.48,0.50,0.45,0.45,0.40,0.40,0.42,0.50,0.62,0.56],
        "RollingAvgFinish_5":          [1.8,2.5,3.1,4.2,3.8,2.9,6.5,5.8,9.1,8.4,
                                        7.2,8.9,11.2,13.5,14.1,15.0,12.8,11.5,7.0,6.5],
        "DNFRate_rolling":             [0.05,0.04,0.07,0.05,0.06,0.04,0.08,0.09,0.10,0.11,
                                        0.09,0.10,0.12,0.13,0.14,0.14,0.12,0.11,0.08,0.07],
        "AdaptationLagFactor":         [1.0,1.0,1.0,0.85,1.0,1.0,0.85,1.0,1.0,1.0,
                                        1.0,1.0,1.0,1.0,1.0,0.80,1.0,0.80,0.80,1.0],
        "SC_Probability_Adjusted":     [0.40]*20,
        "WeatherRiskScore":            [0.12]*20,
    })

    sim   = MonteCarloSimulator(n_sims=5000)
    probs = sim.simulate(dummy)

    print("\n=== 2026 Race Simulation Results ===")
    print(probs[["Driver","Team","WinProb","PodiumProb","Top10Prob",
                 "DNFProb","ExpectedPos","PosStdDev"]].to_string(index=False))
    print("\nPodium matrix:")
    print(sim.get_podium_matrix().to_string(index=False))
