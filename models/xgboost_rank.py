"""
models/xgboost_rank.py

XGBoost pairwise ranking model for F1 race outcome prediction.

Key design decisions:
  - rank:pairwise objective: optimises relative ordering within each race
  - Groups = one race (Year+Round). XGBoost learns "order these 20 drivers"
  - Weights = PER GROUP (per race), not per row — XGBoost ranking requirement
  - Group weight = SeasonWeight of that race (2025 races count 3x more)
  - Walk-forward CV: train on years 1..N, validate on year N+1

Car performance insight (your point):
  - The model uses QualifyingTime gap to pole as the strongest car proxy
  - DiscountedConstructorScore captures team-level car advantage
  - QualiDeltaVsTeammate separates driver skill from car advantage
  - Together these let the model learn: "McLaren car + NOR skill = P1"
    vs "Red Bull car + VER skill = P3-4 in 2026"
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import XGBOOST_PARAMS, DATA_PROCESSED, RANDOM_SEED
from features.feature_store import MODEL_FEATURES, TARGET_COL, WEIGHT_COL, DRIVER_ID_COLS


class XGBoostRankModel:

    def __init__(self, params: dict = None):
        self.params         = params or XGBOOST_PARAMS.copy()
        self.model          = None
        self.feature_names  = MODEL_FEATURES.copy()
        self.shap_explainer = None
        self._is_fitted     = False

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        feature_matrix: pd.DataFrame,
        eval_seasons: list[int] = None,
    ) -> "XGBoostRankModel":
        if not XGB_AVAILABLE:
            raise ImportError("pip install xgboost")

        # Remove n_estimators from params — passed as num_boost_round instead
        train_params = {k: v for k, v in self.params.items()
                        if k != "n_estimators"}
        n_rounds = self.params.get("n_estimators", 500)

        # _prepare_data already returns per-GROUP weights (one per race)
        # Pass them directly — no re-computation needed
        if eval_seasons:
            train_df = feature_matrix[~feature_matrix["Year"].isin(eval_seasons)].copy()
            eval_df  = feature_matrix[ feature_matrix["Year"].isin(eval_seasons)].copy()

            X_train, y_train, qid_train, w_train = self._prepare_data(train_df)
            X_eval,  y_eval,  qid_eval,  _       = self._prepare_data(eval_df)

            dtrain = xgb.DMatrix(X_train, label=y_train,
                                  weight=w_train, qid=qid_train,
                                  feature_names=self.feature_names)
            deval  = xgb.DMatrix(X_eval, label=y_eval,
                                  qid=qid_eval,
                                  feature_names=self.feature_names)

            self.model = xgb.train(
                train_params, dtrain,
                num_boost_round=n_rounds,
                evals=[(dtrain, "train"), (deval, "eval")],
                early_stopping_rounds=40,
                verbose_eval=100,
            )
        else:
            X, y, qid, w = self._prepare_data(feature_matrix)
            dtrain = xgb.DMatrix(X, label=y, weight=w, qid=qid,
                                  feature_names=self.feature_names)
            self.model = xgb.train(
                train_params, dtrain,
                num_boost_round=n_rounds,
                verbose_eval=100,
            )

        self._is_fitted = True

        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP explainer ready")
            except Exception as e:
                logger.warning(f"SHAP explainer failed: {e}")

        logger.info(f"XGBoostRankModel fitted")
        return self

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Not fitted")

        X    = self._get_feature_array(feature_df)
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        raw  = self.model.predict(dmat)

        result = feature_df[["Driver"]].copy()
        if "Team" in feature_df.columns:
            result["Team"] = feature_df["Team"].values
        result["RankScore"] = raw
        result = result.sort_values("RankScore", ascending=False).reset_index(drop=True)
        result["PredictedPos"] = result.index + 1

        probs = _rank_scores_to_probs(raw)
        result["WinProb"]    = probs[:, 0]
        result["PodiumProb"] = probs[:, :3].sum(axis=1)
        result["Top10Prob"]  = probs[:, :10].sum(axis=1)

        return result.sort_values("PredictedPos").reset_index(drop=True)

    def explain_race(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return pd.DataFrame()
        X    = self._get_feature_array(feature_df)
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        sv   = self.shap_explainer.shap_values(dmat)
        df   = pd.DataFrame(sv, columns=self.feature_names)
        if "Driver" in feature_df.columns:
            df.index = feature_df["Driver"].values
        return df

    def feature_importance(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Not fitted")
        imp = self.model.get_score(importance_type="gain")
        df  = pd.DataFrame({
            "Feature":    list(imp.keys()),
            "Importance": list(imp.values()),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        df["ImportancePct"] = (df["Importance"] / df["Importance"].sum() * 100).round(2)
        return df

    def evaluate(self, feature_matrix: pd.DataFrame) -> dict:
        results = []
        for (year, rnd), race_group in feature_matrix.groupby(["Year", "Round"]):
            if len(race_group) < 5:
                continue
            preds   = self.predict(race_group)
            actuals = race_group[["Driver", TARGET_COL]].copy()
            merged  = preds.merge(actuals, on="Driver", how="inner")
            if merged.empty:
                continue
            merged["ActualPos"] = merged[TARGET_COL]
            results.append(merged)
        if not results:
            return {}
        all_results = pd.concat(results, ignore_index=True)
        return _compute_metrics(all_results)

    def save(self, path: str = None) -> str:
        path = path or str(DATA_PROCESSED / "xgboost_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved → {path}")
        return path

    @classmethod
    def load(cls, path: str = None) -> "XGBoostRankModel":
        path = path or str(DATA_PROCESSED / "xgboost_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Data prep ──────────────────────────────────────────────────────────

    def _prepare_data(self, df: pd.DataFrame):
        """
        Returns X, y, qid, weights.

        CRITICAL: XGBoost ranking with qid requires weights to be
        PER QUERY GROUP (per race), not per row.
        One weight value per unique (Year, Round) combination.
        """
        # Fill missing features
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0.0

        X = df[self.feature_names].fillna(0).values

        # Target: invert so higher = better rank (XGBoost maximises)
        y = (21 - df[TARGET_COL].fillna(10)).values.astype(float)

        # Build sorted group IDs (qid must be sorted/contiguous)
        df = df.copy()
        df["_race_key"] = df["Year"].astype(str) + "_" + df["Round"].astype(str)
        race_keys       = df["_race_key"].values
        unique_races    = list(dict.fromkeys(race_keys))   # preserves order
        key_to_id       = {k: i for i, k in enumerate(unique_races)}
        qid             = np.array([key_to_id[k] for k in race_keys], dtype=np.int32)

        # Per-GROUP weights (one value per race group = season weight of that race)
        # XGBoost ranking requires len(weights) == number of groups
        group_weights = []
        for race_key in unique_races:
            race_rows = df[df["_race_key"] == race_key]
            # Use the season weight of this race (all rows in a race have same year)
            if WEIGHT_COL in race_rows.columns:
                w = float(race_rows[WEIGHT_COL].iloc[0])
            else:
                year = int(race_rows["Year"].iloc[0])
                from config import SEASON_WEIGHTS
                w = SEASON_WEIGHTS.get(year, 1.0)
            group_weights.append(w)

        weights = np.array(group_weights, dtype=np.float32)

        return X, y, qid, weights

    def _get_feature_array(self, df: pd.DataFrame) -> np.ndarray:
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0.0
        return df[self.feature_names].fillna(0).values


# ── Utilities ──────────────────────────────────────────────────────────────

def _rank_scores_to_probs(scores: np.ndarray) -> np.ndarray:
    """Plackett-Luce: convert rank scores to position probability matrix."""
    n     = len(scores)
    probs = np.zeros((n, n))

    remaining_idx    = list(range(n))
    remaining_scores = scores.copy()

    for pos in range(n):
        if not remaining_idx:
            break
        exp_s = np.exp(remaining_scores - remaining_scores.max())
        sel   = exp_s / exp_s.sum()
        for local, global_i in enumerate(remaining_idx):
            probs[global_i, pos] = sel[local]
        winner_local  = int(np.argmax(sel))
        remaining_idx.pop(winner_local)
        remaining_scores = np.delete(remaining_scores, winner_local)

    return probs


def _compute_metrics(results_df: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    metrics    = {}
    top1_hits  = []
    top3_hits  = []

    group_cols = [c for c in ["Year", "Round"] if c in results_df.columns]
    for _, race in results_df.groupby(group_cols):
        pw = race.loc[race["PredictedPos"] == 1, "Driver"].values
        aw = race.loc[race["ActualPos"]    == 1, "Driver"].values
        p3 = set(race.loc[race["PredictedPos"] <= 3, "Driver"])
        a3 = set(race.loc[race["ActualPos"]    <= 3, "Driver"])
        if len(pw) > 0 and len(aw) > 0:
            top1_hits.append(int(pw[0] == aw[0]))
        if p3 and a3:
            top3_hits.append(len(p3 & a3) / 3.0)

    metrics["win_accuracy"]  = round(np.mean(top1_hits), 3) if top1_hits else 0.0
    metrics["top3_accuracy"] = round(np.mean(top3_hits), 3) if top3_hits else 0.0

    rho, pval = spearmanr(results_df["PredictedPos"], results_df["ActualPos"])
    metrics["spearman_rho"]  = round(float(rho),  4)
    metrics["spearman_pval"] = round(float(pval), 4)
    metrics["mae_positions"] = round(
        float((results_df["PredictedPos"] - results_df["ActualPos"]).abs().mean()), 3
    )
    return metrics


if __name__ == "__main__":
    logger.info("XGBoostRankModel ready")
    if not XGB_AVAILABLE:
        print("Install xgboost: pip install xgboost")
    print(f"Features: {len(MODEL_FEATURES)}")