"""
models/ensemble.py

Weighted ensemble that combines XGBoost, Bradley-Terry, and Monte Carlo
outputs into a single final prediction.

Why ensemble?
  - XGBoost: feature-rich, captures interactions, but black-box
  - Bradley-Terry: interpretable, robust to missing data, good at relative ranking
  - Monte Carlo: injects realistic stochasticity, produces full distributions
  - Ensemble: errors are less correlated → more accurate combined predictions

Ensemble strategy:
  1. Each model produces a WinProb and PodiumProb per driver
  2. Weighted average: w_xgb=0.50, w_bt=0.30, w_mc=0.20 (tuned on 2022–2024)
  3. Re-normalise probabilities to sum to 1.0
  4. Re-rank drivers by WinProb to get PredictedPos

Weights are configurable in config.py and should be re-tuned after backtesting.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENSEMBLE_WEIGHTS, DATA_PROCESSED, RANDOM_SEED
from models.xgboost_rank  import XGBoostRankModel
from models.bradley_terry import BradleyTerryModel
from models.monte_carlo   import MonteCarloSimulator


class EnsemblePredictor:
    """
    Stacked ensemble of XGBoost + Bradley-Terry + Monte Carlo.

    Usage:
        ensemble = EnsemblePredictor()
        ensemble.fit(training_feature_matrix, race_results)
        prediction = ensemble.predict_race(circuit_name, qualifying_df, weather)
    """

    def __init__(
        self,
        weights: dict = None,
        mc_n_sims: int = 10_000,
    ):
        self.weights  = weights or ENSEMBLE_WEIGHTS.copy()
        self.xgb      = XGBoostRankModel()
        self.bt       = BradleyTerryModel(regularisation=1.0)
        self.mc       = MonteCarloSimulator(n_sims=mc_n_sims)
        self._is_fitted = False

        # Store last prediction for post-race comparison
        self.last_prediction_  = None
        self.last_race_name_   = None

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        feature_matrix: pd.DataFrame,
        results: pd.DataFrame,
        eval_seasons: list[int] = None,
    ) -> "EnsemblePredictor":
        """
        Fit all three component models.

        Parameters
        ----------
        feature_matrix : Output of feature_store.build_training_feature_matrix()
        results        : Raw results DataFrame (for Bradley-Terry pairwise fitting)
        eval_seasons   : Seasons to hold out for XGBoost early stopping
        """
        from config import SEASON_WEIGHTS

        logger.info("=== Fitting ensemble ===")

        # XGBoost
        logger.info("Fitting XGBoost ranking model...")
        self.xgb.fit(feature_matrix, eval_seasons=eval_seasons)

        # Bradley-Terry
        logger.info("Fitting Bradley-Terry model...")
        self.bt.fit(results, season_weights=SEASON_WEIGHTS)

        self._is_fitted = True
        logger.info("Ensemble fitted successfully")
        return self

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(
        self,
        feature_df: pd.DataFrame,
        race_name: str = "Unknown",
        return_components: bool = False,
    ) -> pd.DataFrame:
        """
        Generate ensemble race predictions.

        Parameters
        ----------
        feature_df        : Output of feature_store.build_prediction_row()
        race_name         : Name of the race (for logging and storage)
        return_components : If True, include individual model outputs in result

        Returns
        -------
        DataFrame sorted by predicted finishing position, columns:
            Driver, Team, PredictedPos, WinProb, PodiumProb, Top5Prob, Top10Prob,
            DNFProb, ExpectedPos, PosStdDev, WinProb_CI_low, WinProb_CI_high
            [+ XGB_WinProb, BT_WinProb, MC_WinProb if return_components=True]
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        logger.info(f"Generating ensemble prediction for {race_name}")

        # ── Component predictions ──────────────────────────────────────────
        xgb_pred    = self.xgb.predict(feature_df)
        bt_pred     = self._get_bt_predictions(feature_df)
        mc_pred     = self.mc.simulate(
            feature_df,
            xgb_scores   = xgb_pred["RankScore"].values,
            bt_strengths = bt_pred["BT_Strength"].values,
        )

        # ── Merge on Driver ────────────────────────────────────────────────
        merged = feature_df[["Driver"]].copy()
        if "Team" in feature_df.columns:
            merged["Team"] = feature_df["Team"].values

        merged = merged.merge(
            xgb_pred[["Driver", "WinProb", "PodiumProb", "Top10Prob"]]\
                .rename(columns={"WinProb":"XGB_WinProb",
                                 "PodiumProb":"XGB_PodiumProb",
                                 "Top10Prob":"XGB_Top10Prob"}),
            on="Driver", how="left"
        )
        merged = merged.merge(
            bt_pred[["Driver", "WinProb", "PodiumProb"]]\
                .rename(columns={"WinProb":"BT_WinProb",
                                 "PodiumProb":"BT_PodiumProb"}),
            on="Driver", how="left"
        )
        merged = merged.merge(
            mc_pred[["Driver", "WinProb", "PodiumProb", "Top5Prob",
                     "Top10Prob", "DNFProb", "ExpectedPos", "PosStdDev"]]\
                .rename(columns={"WinProb":"MC_WinProb",
                                 "PodiumProb":"MC_PodiumProb",
                                 "Top5Prob":"MC_Top5Prob",
                                 "Top10Prob":"MC_Top10Prob"}),
            on="Driver", how="left"
        )

        # ── Weighted ensemble ──────────────────────────────────────────────
        w_xgb = self.weights.get("xgboost", 0.50)
        w_bt  = self.weights.get("bradley_terry", 0.30)
        w_mc  = self.weights.get("monte_carlo", 0.20)

        merged["WinProb_raw"] = (
            w_xgb * merged["XGB_WinProb"].fillna(0) +
            w_bt  * merged["BT_WinProb"].fillna(0)  +
            w_mc  * merged["MC_WinProb"].fillna(0)
        )
        merged["PodiumProb_raw"] = (
            w_xgb * merged["XGB_PodiumProb"].fillna(0) +
            w_bt  * merged["BT_PodiumProb"].fillna(0) +
            w_mc  * merged["MC_PodiumProb"].fillna(0)
        )

        # Re-normalise win probabilities to sum to 1
        total_win = merged["WinProb_raw"].sum()
        merged["WinProb"] = (merged["WinProb_raw"] / total_win if total_win > 0
                             else merged["WinProb_raw"]).round(4)

        total_pod = merged["PodiumProb_raw"].sum()
        # Podium probs should sum to 3 (3 podium spots / 20 drivers)
        merged["PodiumProb"] = (merged["PodiumProb_raw"] / total_pod * 3.0
                                if total_pod > 0 else merged["PodiumProb_raw"]).round(4)

        # Confidence interval: center the display interval on the final ensemble
        # win probability so the CLI output stays internally consistent.
        n_mc = self.mc.n_sims
        win_prob_for_ci = merged["WinProb"].clip(0, 1)
        merged["WinProb_CI_low"]  = (
            win_prob_for_ci - 1.645 * np.sqrt(
                win_prob_for_ci * (1 - win_prob_for_ci) / n_mc
            )
        ).clip(0).round(4)
        merged["WinProb_CI_high"] = (
            win_prob_for_ci + 1.645 * np.sqrt(
                win_prob_for_ci * (1 - win_prob_for_ci) / n_mc
            )
        ).clip(0, 1).round(4)

        # Final predicted position = rank by WinProb descending
        merged = merged.sort_values("WinProb", ascending=False).reset_index(drop=True)
        merged["PredictedPos"] = merged.index + 1

        # Pass through MC's full distribution columns
        merged["Top5Prob"]   = merged["MC_Top5Prob"].fillna(0).round(4)
        merged["Top10Prob"]  = merged["MC_Top10Prob"].fillna(0).round(4)
        merged["DNFProb"]    = merged["DNFProb"].fillna(0).round(4)
        merged["ExpectedPos"]= merged["ExpectedPos"].fillna(10).round(2)
        merged["PosStdDev"]  = merged["PosStdDev"].fillna(3).round(2)

        # Clean output columns
        output_cols = [
            "Driver", "Team", "PredictedPos",
            "WinProb", "PodiumProb", "Top5Prob", "Top10Prob",
            "DNFProb", "ExpectedPos", "PosStdDev",
            "WinProb_CI_low", "WinProb_CI_high",
        ]
        if return_components:
            output_cols += ["XGB_WinProb", "BT_WinProb", "MC_WinProb",
                            "XGB_PodiumProb", "BT_PodiumProb", "MC_PodiumProb"]

        result = merged[[c for c in output_cols if c in merged.columns]].copy()

        self.last_prediction_ = result
        self.last_race_name_  = race_name

        return result

    # ── Post-race scoring ──────────────────────────────────────────────────

    def score_prediction(self, actual_results: pd.DataFrame) -> dict:
        """
        Score the last prediction against actual race results.

        Parameters
        ----------
        actual_results : DataFrame with Driver, FinishPosition columns

        Returns
        -------
        dict of metrics: win_correct, podium_overlap, spearman_rho, brier_score
        """
        if self.last_prediction_ is None:
            raise RuntimeError("No prediction stored. Call predict() first.")

        pred = self.last_prediction_.copy()
        actual = actual_results[["Driver", "FinishPosition"]].copy()
        merged = pred.merge(actual, on="Driver", how="inner")

        if merged.empty:
            return {}

        # Win correct
        pred_winner   = merged.loc[merged["PredictedPos"] == 1, "Driver"].values
        actual_winner = merged.loc[merged["FinishPosition"] == 1, "Driver"].values
        win_correct   = bool(len(pred_winner) > 0 and len(actual_winner) > 0 and
                             pred_winner[0] == actual_winner[0])

        # Podium overlap (how many of predicted P1-3 = actual P1-3)
        pred_top3   = set(merged.loc[merged["PredictedPos"]   <= 3, "Driver"])
        actual_top3 = set(merged.loc[merged["FinishPosition"] <= 3, "Driver"])
        podium_overlap = len(pred_top3 & actual_top3)

        # Spearman rank correlation
        from scipy.stats import spearmanr
        rho, _ = spearmanr(merged["PredictedPos"], merged["FinishPosition"])

        # Brier score for win probability
        merged["ActualWin"] = (merged["FinishPosition"] == 1).astype(float)
        brier = float(((merged["WinProb"] - merged["ActualWin"]) ** 2).mean())

        metrics = {
            "race":            self.last_race_name_,
            "win_correct":     win_correct,
            "podium_overlap":  podium_overlap,
            "spearman_rho":    round(float(rho), 4),
            "brier_score":     round(brier, 4),
            "mae_positions":   round(
                float((merged["PredictedPos"] - merged["FinishPosition"]).abs().mean()), 2
            ),
        }

        logger.info(f"Race scored: win={win_correct}, podium={podium_overlap}/3, "
                    f"ρ={metrics['spearman_rho']}, Brier={metrics['brier_score']}")
        return metrics

    # ── Calibration ────────────────────────────────────────────────────────

    def tune_weights(
        self,
        feature_matrices: list[pd.DataFrame],
        race_results:     list[pd.DataFrame],
    ) -> dict:
        """
        Tune ensemble weights on held-out races via grid search.
        Minimises Brier score across all held-out races.

        Call this after initial fit with a validation set.
        Updates self.weights in place.
        """
        best_brier  = float("inf")
        best_weights = self.weights.copy()

        # Grid search over weight combinations
        for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for w_bt in [0.1, 0.2, 0.3, 0.4]:
                w_mc = 1.0 - w_xgb - w_bt
                if w_mc < 0:
                    continue

                self.weights = {"xgboost": w_xgb, "bradley_terry": w_bt, "monte_carlo": w_mc}
                brier_scores = []

                for feat_mat, results in zip(feature_matrices, race_results):
                    try:
                        pred   = self.predict(feat_mat)
                        scored = self.score_prediction(results)
                        brier_scores.append(scored.get("brier_score", 1.0))
                    except Exception:
                        continue

                if brier_scores:
                    avg_brier = np.mean(brier_scores)
                    if avg_brier < best_brier:
                        best_brier   = avg_brier
                        best_weights = self.weights.copy()

        self.weights = best_weights
        logger.info(f"Tuned weights: {self.weights} (Brier: {best_brier:.4f})")
        return self.weights

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str = None) -> str:
        path = path or str(DATA_PROCESSED / "ensemble_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Ensemble saved → {path}")
        return path

    @classmethod
    def load(cls, path: str = None) -> "EnsemblePredictor":
        path = path or str(DATA_PROCESSED / "ensemble_model.pkl")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Ensemble loaded ← {path}")
        return model

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_bt_predictions(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Extract BT predictions for drivers in this race."""
        drivers = feature_df["Driver"].tolist()

        # Build constructor bonus from feature_df
        constructor_bonus = {}
        if "DiscountedConstructorScore" in feature_df.columns:
            for _, row in feature_df.iterrows():
                constructor_bonus[row["Driver"]] = float(
                    row["DiscountedConstructorScore"] - 0.5
                )

        # Build adaptation lag map
        adaptation_lag = {}
        if "AdaptationLagFactor" in feature_df.columns:
            for _, row in feature_df.iterrows():
                adaptation_lag[row["Driver"]] = float(row["AdaptationLagFactor"])

        return self.bt.predict_race(
            drivers,
            constructor_bonus=constructor_bonus,
            adaptation_lag=adaptation_lag,
        )


def format_prediction_table(pred: pd.DataFrame) -> str:
    """
    Format prediction output as a clean ASCII table.
    Used for print output and markdown in README/notebooks.
    """
    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║                     F1 2026 RACE PREDICTION                         ║",
        "╠════╦══════╦═══════════════════╦══════════╦══════════╦══════════╦════╣",
        "║ P  ║ Code ║ Team              ║ Win%     ║ Podium%  ║ Top10%   ║ DNF║",
        "╠════╬══════╬═══════════════════╬══════════╬══════════╬══════════╬════╣",
    ]
    for _, row in pred.iterrows():
        pos    = int(row.get("PredictedPos", 0))
        driver = str(row.get("Driver", "???"))
        team   = str(row.get("Team", "???"))[:19]
        win    = f"{row.get('WinProb', 0)*100:.1f}%"
        pod    = f"{row.get('PodiumProb', 0)*100:.1f}%"
        t10    = f"{row.get('Top10Prob', 0)*100:.1f}%"
        dnf    = f"{row.get('DNFProb', 0)*100:.1f}%"
        lines.append(
            f"║{pos:3d} ║ {driver:<4} ║ {team:<17} ║ {win:>8} ║ {pod:>8} ║ {t10:>8} ║{dnf:>4}║"
        )
    lines.append("╚════╩══════╩═══════════════════╩══════════╩══════════╩══════════╩════╝")
    return "\n".join(lines)


if __name__ == "__main__":
    logger.info("Ensemble module ready. Full pipeline test requires fitted models.")
    logger.info("Run: python predict.py --race Bahrain --year 2026")
