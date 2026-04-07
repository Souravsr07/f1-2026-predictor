"""
models/bradley_terry.py

Bradley-Terry pairwise ranking model for F1 predictions.

Why Bradley-Terry alongside XGBoost?
  - XGBoost needs a full feature vector per driver — missing data = imputed 0s
  - Bradley-Terry learns from pairwise comparisons directly:
    "In the 2024 Bahrain GP, VER finished ahead of NOR. Given what we know
     about their relative strengths, update our estimate of P(VER beats NOR)."
  - Handles rookies and team switchers more gracefully: we just have wider
    uncertainty intervals for them, rather than forcing an imputed feature
  - Acts as a strong ensemble component — structurally different from XGBoost
    so errors are less correlated

Model:
  Each driver i has a latent strength parameter β_i ∈ ℝ.
  P(i beats j) = sigmoid(β_i - β_j) = 1 / (1 + exp(-(β_i - β_j)))

  Parameters estimated by maximising log-likelihood over all historical
  pairwise outcomes (i finished ahead of j in race r).

  For 2026 prediction:
  - Apply λ-discounted constructor bonus to β_i
  - Apply adaptation lag penalty to β_i for team switchers/rookies
  - P(full ordering) estimated via Plackett-Luce extension of B-T
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit   # sigmoid
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, RANDOM_SEED, DRIVER_TEAM_2026
from features.feature_store import TARGET_COL


class BradleyTerryModel:
    """
    Bradley-Terry pairwise ranking model.

    Attributes
    ----------
    strengths_    : dict[driver -> beta (float)] — fitted strength parameters
    uncertainty_  : dict[driver -> sigma (float)] — parameter uncertainty
    driver_list_  : list of drivers in the fitted model
    """

    def __init__(self, regularisation: float = 1.0):
        """
        Parameters
        ----------
        regularisation : L2 regularisation on beta params.
                         Higher = more shrinkage toward 0 (mean reversion).
                         Use higher values for 2026 to encode reg-reset uncertainty.
        """
        self.regularisation = regularisation
        self.strengths_     = {}
        self.uncertainty_   = {}
        self.driver_list_   = []
        self._is_fitted     = False

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        results: pd.DataFrame,
        season_weights: dict[int, float] = None,
    ) -> "BradleyTerryModel":
        """
        Fit Bradley-Terry model from race results.

        Parameters
        ----------
        results        : DataFrame with Year, Round, Driver, FinishPosition, Team
        season_weights : dict{year: weight}. Recent seasons weighted higher.

        Returns
        -------
        self (fitted)
        """
        logger.info("Fitting Bradley-Terry model...")

        # Build all pairwise comparisons
        pairs = self._build_pairs(results, season_weights)
        if pairs.empty:
            raise ValueError("No pairwise comparisons extracted from results")

        logger.info(f"Pairwise comparisons: {len(pairs)} pairs from "
                    f"{results['Driver'].nunique()} drivers")

        # Driver index
        self.driver_list_ = sorted(results["Driver"].unique())
        driver_to_idx     = {d: i for i, d in enumerate(self.driver_list_)}
        n_drivers         = len(self.driver_list_)

        # Initial parameters (all zeros — equal strength)
        beta0 = np.zeros(n_drivers)

        # Optimise negative log-likelihood
        winner_idx = pairs["winner"].map(driver_to_idx).values
        loser_idx  = pairs["loser"].map(driver_to_idx).values
        weights    = pairs["weight"].values

        def neg_log_likelihood(beta):
            # For each pair: log P(winner beats loser) = log sigmoid(beta_w - beta_l)
            diffs     = beta[winner_idx] - beta[loser_idx]
            log_probs = np.log(expit(diffs) + 1e-10)
            nll       = -np.sum(weights * log_probs)
            # L2 regularisation
            nll      += 0.5 * self.regularisation * np.sum(beta ** 2)
            return nll

        def grad_nll(beta):
            diffs    = beta[winner_idx] - beta[loser_idx]
            sigmoid_ = expit(diffs)
            error    = weights * (sigmoid_ - 1.0)   # residual

            grad = np.zeros(n_drivers)
            np.add.at(grad, winner_idx, error)
            np.add.at(grad, loser_idx, -error)
            grad += self.regularisation * beta
            return grad

        result = minimize(
            neg_log_likelihood,
            beta0,
            jac=grad_nll,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning(f"Optimisation did not fully converge: {result.message}")

        beta_fitted = result.x

        # Normalise: centre at 0
        beta_fitted -= beta_fitted.mean()

        # Store strengths
        self.strengths_ = {
            d: float(beta_fitted[i]) for i, d in enumerate(self.driver_list_)
        }

        # Approximate uncertainty via diagonal of inverse Hessian
        # (simplified: use 1/sqrt(n_comparisons_per_driver))
        driver_counts = pd.concat([pairs["winner"], pairs["loser"]]).value_counts()
        for d in self.driver_list_:
            n = driver_counts.get(d, 1)
            self.uncertainty_[d] = float(1.0 / np.sqrt(n))

        self._is_fitted = True
        logger.info(f"Bradley-Terry fitted. Top 5 by strength: "
                    f"{sorted(self.strengths_.items(), key=lambda x: -x[1])[:5]}")
        return self

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict_race(
        self,
        drivers: list[str],
        constructor_bonus: dict[str, float] = None,
        adaptation_lag:   dict[str, float] = None,
    ) -> pd.DataFrame:
        """
        Predict finishing positions for a race.

        Parameters
        ----------
        drivers          : List of driver codes in the race
        constructor_bonus: dict{driver: bonus} — λ-discounted constructor score
                           added to beta (scaled). None = use raw strengths.
        adaptation_lag   : dict{driver: lag_factor} — multiplied onto beta.
                           Values < 1.0 shrink strength toward 0.

        Returns
        -------
        DataFrame: Driver, BT_Strength, WinProb, PodiumProb, Top10Prob, PredictedPos
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        rows = []
        for driver in drivers:
            beta = self.strengths_.get(driver, 0.0)

            # Apply constructor bonus (scale factor 0.5 — don't let it dominate)
            if constructor_bonus and driver in constructor_bonus:
                beta += 0.5 * constructor_bonus[driver]

            # Apply adaptation lag: shrink strength toward 0
            if adaptation_lag and driver in adaptation_lag:
                lag  = adaptation_lag[driver]
                beta = beta * lag   # lag < 1 → weaker strength

            rows.append({"Driver": driver, "BT_Strength": beta})

        bt_df = pd.DataFrame(rows)

        # Plackett-Luce win probabilities from strengths
        probs = self._plackett_luce_probs(bt_df["BT_Strength"].values)

        bt_df["WinProb"]    = probs[:, 0]
        bt_df["PodiumProb"] = probs[:, :3].sum(axis=1)
        bt_df["Top10Prob"]  = probs[:, :10].sum(axis=1)
        bt_df["PredictedPos"] = bt_df["WinProb"].rank(ascending=False).astype(int)

        return bt_df.sort_values("PredictedPos").reset_index(drop=True)

    def predict_head_to_head(self, driver_a: str, driver_b: str) -> float:
        """
        Return P(driver_a finishes ahead of driver_b).
        Pure Bradley-Terry pairwise probability.
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted")

        beta_a = self.strengths_.get(driver_a, 0.0)
        beta_b = self.strengths_.get(driver_b, 0.0)
        return float(expit(beta_a - beta_b))

    def get_strength_table(self) -> pd.DataFrame:
        """Return ranked driver strengths with uncertainty."""
        rows = [
            {"Driver": d, "BT_Strength": b, "Uncertainty": self.uncertainty_.get(d, 0.1)}
            for d, b in self.strengths_.items()
        ]
        df = pd.DataFrame(rows).sort_values("BT_Strength", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        return df

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, results: pd.DataFrame) -> dict:
        """
        Evaluate pairwise prediction accuracy on held-out results.

        Returns:
            pairwise_accuracy : fraction of pairwise comparisons correctly ranked
            win_accuracy      : fraction of races where predicted winner = actual
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted")

        pairs = self._build_pairs(results)
        if pairs.empty:
            return {}

        correct = 0
        total   = 0
        for _, row in pairs.iterrows():
            w = row["winner"]
            l = row["loser"]
            p = self.predict_head_to_head(w, l)
            correct += int(p > 0.5)
            total   += 1

        pairwise_acc = correct / total if total > 0 else 0.0

        # Win accuracy
        win_hits = []
        for (year, rnd), race in results.groupby(["Year", "Round"]):
            drivers      = race["Driver"].tolist()
            preds        = self.predict_race(drivers)
            pred_winner  = preds.iloc[0]["Driver"]
            actual_winner = race.loc[race[TARGET_COL] == 1, "Driver"].values
            if len(actual_winner) > 0:
                win_hits.append(int(pred_winner == actual_winner[0]))

        return {
            "pairwise_accuracy": round(pairwise_acc, 4),
            "win_accuracy":      round(np.mean(win_hits), 4) if win_hits else 0.0,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str = None) -> str:
        path = path or str(DATA_PROCESSED / "bradley_terry_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"BT model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str = None) -> "BradleyTerryModel":
        path = path or str(DATA_PROCESSED / "bradley_terry_model.pkl")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"BT model loaded ← {path}")
        return model

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_pairs(
        self,
        results: pd.DataFrame,
        season_weights: dict[int, float] = None,
    ) -> pd.DataFrame:
        """
        Extract all pairwise (winner, loser) comparisons from race results.
        Within each race, every pair (i, j) where i finished ahead of j
        contributes one row: winner=i, loser=j, weight=season_weight.
        """
        pairs = []
        for (year, rnd), race in results.groupby(["Year", "Round"]):
            race = race.dropna(subset=[TARGET_COL])
            race = race[race[TARGET_COL] > 0].sort_values(TARGET_COL)
            drivers = race["Driver"].tolist()

            w = 1.0
            if season_weights and year in season_weights:
                w = season_weights[year]

            # Only top-N pairs to keep computation manageable
            # (full N×N is O(N²) pairs per race — 20 drivers = 190 pairs)
            n = len(drivers)
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append({
                        "winner": drivers[i],
                        "loser":  drivers[j],
                        "weight": w,
                    })

        return pd.DataFrame(pairs)

    @staticmethod
    def _plackett_luce_probs(strengths: np.ndarray) -> np.ndarray:
        """
        Compute Plackett-Luce position probabilities from strength parameters.
        probs[i, k] = P(driver i finishes in position k+1).

        Uses the recursive formula:
          P(i in pos 1) = exp(β_i) / Σ exp(β_j)
          P(i in pos 2 | not in pos 1) = sum over who won P(j wins) × P(i | remaining)
          etc.

        For efficiency: Monte Carlo approximation with N=2000 simulations.
        """
        n      = len(strengths)
        n_sims = 2000
        counts = np.zeros((n, n))

        rng = np.random.default_rng(RANDOM_SEED)
        for _ in range(n_sims):
            remaining = list(range(n))
            rem_strengths = strengths.copy()
            for pos in range(n):
                exp_s = np.exp(rem_strengths - rem_strengths.max())
                probs = exp_s / exp_s.sum()
                chosen_local = rng.choice(len(remaining), p=probs)
                chosen_global = remaining[chosen_local]
                counts[chosen_global, pos] += 1
                remaining.pop(chosen_local)
                rem_strengths = np.delete(rem_strengths, chosen_local)

        return counts / n_sims


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    import pandas as pd

    rng = np.random.default_rng(42)
    drivers = ["VER", "NOR", "LEC", "HAM", "RUS", "PIA", "SAI", "ALO",
               "STR", "GAS", "ALB", "TSU", "OCO", "BEA", "HUL", "BOR",
               "DOO", "HAD", "ANT", "LAW"]

    synthetic_results = []
    for year in [2023, 2024, 2025]:
        for rnd in range(1, 23):
            order = rng.permutation(drivers)
            for pos, driver in enumerate(order, 1):
                synthetic_results.append({
                    "Year": year, "Round": rnd,
                    "Driver": driver, "FinishPosition": pos,
                })

    df = pd.DataFrame(synthetic_results)

    from config import SEASON_WEIGHTS
    bt = BradleyTerryModel(regularisation=1.0)
    bt.fit(df, season_weights=SEASON_WEIGHTS)

    print("\nBradley-Terry strength table:")
    print(bt.get_strength_table().head(10).to_string())

    print("\nHead-to-head: VER vs NOR:", round(bt.predict_head_to_head("VER", "NOR"), 3))

    preds = bt.predict_race(drivers)
    print("\nPredicted race:")
    print(preds[["Driver", "BT_Strength", "WinProb", "PodiumProb"]].to_string())