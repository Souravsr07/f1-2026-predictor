"""
visualisations.py

All chart generation for the F1 2026 predictor.
Every plot function:
  - Takes a prediction/results DataFrame
  - Saves a PNG to results/plots/
  - Returns the figure (for notebook use)
  - Is independently callable — no hidden state

Chart inventory:
  1. win_probability_chart      — bar chart with 90% CI bands
  2. podium_heatmap             — driver × P1/P2/P3 grid
  3. position_distribution      — violin/box per driver
  4. pre_post_race_overlay      — predicted vs actual comparison
  5. shap_waterfall             — feature contributions for one driver
  6. feature_importance_bar     — global model feature importance
  7. championship_trajectory    — rolling points + WDC probability
  8. circuit_dna_radar          — circuit fingerprint spider chart
  9. model_accuracy_tracker     — cumulative accuracy across races
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns

matplotlib.use("Agg")   # non-interactive backend — safe for CI/server

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROCESSED, DRIVER_TEAM_2026

# ── Output directory ───────────────────────────────────────────────────────
PLOTS_DIR = DATA_PROCESSED.parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── F1 team colours (2026 mapping) ────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull":       "#3671C6",
    "McLaren":        "#FF8000",
    "Ferrari":        "#E8002D",
    "Mercedes":       "#27F4D2",
    "Aston Martin":   "#229971",
    "Alpine":         "#FF87BC",
    "Williams":       "#64C4FF",
    "Racing Bulls":   "#6692FF",
    "Haas":           "#B6BABD",
    "Audi":           "#52E252",
    "Cadillac":       "#9B0000",
}

DRIVER_TEAM = DRIVER_TEAM_2026.copy()

# ── Style helper ───────────────────────────────────────────────────────────

def _set_style():
    """Apply consistent dark-minimal style to all charts."""
    plt.rcParams.update({
        "figure.facecolor":   "#0F0F0F",
        "axes.facecolor":     "#0F0F0F",
        "axes.edgecolor":     "#333333",
        "axes.labelcolor":    "#CCCCCC",
        "xtick.color":        "#999999",
        "ytick.color":        "#999999",
        "text.color":         "#CCCCCC",
        "grid.color":         "#222222",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "font.family":        "monospace",
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         150,
    })

def _driver_color(driver: str) -> str:
    team = DRIVER_TEAM.get(driver, "Unknown")
    return TEAM_COLORS.get(team, "#888888")

def _save(fig: plt.Figure, filename: str) -> Path:
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ── 1. Win probability chart ───────────────────────────────────────────────

def win_probability_chart(
    prediction: pd.DataFrame,
    race_name: str,
    year: int = 2026,
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of win probabilities with 90% CI bands.
    Bars coloured by team. Confidence interval from Monte Carlo spread.
    """
    _set_style()
    df = prediction.sort_values("WinProb", ascending=True).tail(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [_driver_color(d) for d in df["Driver"]]
    bars   = ax.barh(df["Driver"], df["WinProb"] * 100, color=colors,
                     height=0.65, alpha=0.90)

    # Confidence intervals
    if "WinProb_CI_low" in df.columns and "WinProb_CI_high" in df.columns:
        for _, row in df.iterrows():
            ax.plot(
                [row["WinProb_CI_low"] * 100, row["WinProb_CI_high"] * 100],
                [row["Driver"], row["Driver"]],
                color="white", alpha=0.5, linewidth=2.5,
                solid_capstyle="round",
            )

    # Value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", ha="left",
                color="#CCCCCC", fontsize=9)

    ax.set_xlabel("Win probability (%)")
    ax.set_title(f"{year} {race_name} GP — win probability", pad=12)
    ax.set_xlim(0, df["WinProb"].max() * 100 * 1.25)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Team legend
    seen_teams = {}
    for d in df["Driver"]:
        t = DRIVER_TEAM.get(d, "Unknown")
        if t not in seen_teams:
            seen_teams[t] = _driver_color(d)
    patches = [mpatches.Patch(color=c, label=t) for t, c in seen_teams.items()]
    ax.legend(handles=patches, loc="lower right", framealpha=0.2, fontsize=8)

    fig.tight_layout()
    if save:
        path = _save(fig, f"win_prob_{race_name.lower().replace(' ', '_')}_{year}.png")
        return fig, path
    return fig


# ── 2. Podium heatmap ──────────────────────────────────────────────────────

def podium_heatmap(
    prediction: pd.DataFrame,
    race_name: str,
    year: int = 2026,
    top_n: int = 14,
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap of P1/P2/P3/Podium probabilities — driver × position.
    The single best chart for LinkedIn posts.
    """
    _set_style()
    df = prediction.sort_values("WinProb", ascending=False).head(top_n).copy()

    prob_cols = ["WinProb", "P2Prob", "P3Prob", "PodiumProb"]
    available = [c for c in prob_cols if c in df.columns]
    labels    = {"WinProb": "P1 win", "P2Prob": "P2", "P3Prob": "P3", "PodiumProb": "Podium"}

    matrix = df[["Driver"] + available].set_index("Driver")[available].copy()
    matrix.columns = [labels.get(c, c) for c in available]
    matrix = matrix * 100   # to percent

    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="#1a1a1a",
        cbar_kws={"label": "Probability (%)", "shrink": 0.6},
        annot_kws={"size": 10},
    )

    # Colour driver labels by team
    for tick in ax.get_yticklabels():
        driver = tick.get_text()
        tick.set_color(_driver_color(driver))

    ax.set_title(f"{year} {race_name} GP — podium probability matrix", pad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    if save:
        path = _save(fig, f"podium_heatmap_{race_name.lower().replace(' ', '_')}_{year}.png")
        return fig, path
    return fig


# ── 3. Position distribution ───────────────────────────────────────────────

def position_distribution(
    prediction: pd.DataFrame,
    race_name: str,
    year: int = 2026,
    top_n: int = 12,
    save: bool = True,
) -> plt.Figure:
    """
    Stacked probability bars showing full P1..P20 distribution per driver.
    Shows the spread (uncertainty) not just the expected position.
    """
    _set_style()
    df = prediction.sort_values("WinProb", ascending=False).head(top_n).copy()

    # Pull out P1..P10 probability columns
    pos_cols = [f"P{i}_prob" for i in range(1, 11) if f"P{i}_prob" in df.columns]
    if not pos_cols:
        # Fallback: use ExpectedPos and PosStdDev to simulate
        return _position_bar_fallback(df, race_name, year, save)

    n_drivers = len(df)
    n_pos     = len(pos_cols)
    matrix    = df[pos_cols].values * 100   # to percent

    pos_colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, n_pos))

    fig, ax = plt.subplots(figsize=(12, 7))

    bottom = np.zeros(n_drivers)
    for i, col in enumerate(pos_cols):
        vals = df[col].values * 100
        ax.bar(
            range(n_drivers),
            vals,
            bottom=bottom,
            color=pos_colors[i],
            width=0.7,
            label=f"P{i+1}",
        )
        bottom += vals

    ax.set_xticks(range(n_drivers))
    ax.set_xticklabels(df["Driver"].values, fontsize=10)
    for tick, driver in zip(ax.get_xticklabels(), df["Driver"]):
        tick.set_color(_driver_color(driver))

    ax.set_ylabel("Probability (%)")
    ax.set_title(f"{year} {race_name} GP — position distribution (top 10 positions)", pad=12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    legend = ax.legend(
        title="Position", ncol=5, loc="upper right",
        framealpha=0.2, fontsize=8
    )

    fig.tight_layout()
    if save:
        path = _save(fig, f"pos_dist_{race_name.lower().replace(' ', '_')}_{year}.png")
        return fig, path
    return fig


def _position_bar_fallback(df, race_name, year, save):
    """Fallback when per-position probs not available: show ExpectedPos with std dev."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = [_driver_color(d) for d in df["Driver"]]
    ax.barh(
        df["Driver"],
        df.get("ExpectedPos", range(1, len(df)+1)),
        xerr=df.get("PosStdDev", 2.0),
        color=colors, height=0.65, alpha=0.85,
        error_kw={"ecolor": "white", "alpha": 0.5, "capsize": 3}
    )
    ax.invert_xaxis()
    ax.set_xlabel("Expected finishing position")
    ax.set_title(f"{year} {race_name} GP — expected position ± uncertainty")
    fig.tight_layout()
    if save:
        path = _save(fig, f"pos_dist_{race_name.lower().replace(' ', '_')}_{year}.png")
        return fig, path
    return fig


# ── 4. Pre/post race overlay ───────────────────────────────────────────────

def pre_post_race_overlay(
    prediction: pd.DataFrame,
    actual: pd.DataFrame,
    race_name: str,
    year: int = 2026,
    save: bool = True,
) -> plt.Figure:
    """
    Side-by-side comparison: predicted order vs actual finishing order.
    Lines connect the same driver between panels — crossing lines = wrong prediction.
    This is the hero chart for post-race LinkedIn posts.
    """
    _set_style()

    merged = prediction[["Driver", "PredictedPos"]].merge(
        actual[["Driver", "FinishPosition"]],
        on="Driver", how="inner"
    ).sort_values("FinishPosition")

    n   = len(merged)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10),
                                    gridspec_kw={"width_ratios": [1, 1]})

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(n + 0.5, 0.5)
        ax.axis("off")

    ax1.set_title("Predicted", fontsize=13, color="#CCCCCC", pad=8)
    ax2.set_title("Actual", fontsize=13, color="#CCCCCC", pad=8)

    pred_sorted   = merged.sort_values("PredictedPos")
    actual_sorted = merged.sort_values("FinishPosition")

    # Draw driver names on each side
    pred_y  = {row["Driver"]: (idx + 1) for idx, (_, row) in enumerate(pred_sorted.iterrows())}
    actual_y = {row["Driver"]: (idx + 1) for idx, (_, row) in enumerate(actual_sorted.iterrows())}

    for driver in merged["Driver"]:
        c  = _driver_color(driver)
        py = pred_y[driver]
        ay = actual_y[driver]

        ax1.text(0.85, py, driver, ha="right", va="center",
                 color=c, fontsize=10, fontweight="bold")
        ax2.text(0.15, ay, driver, ha="left",  va="center",
                 color=c, fontsize=10, fontweight="bold")

        # Connecting line in fig coords
        fig.add_artist(
            matplotlib.lines.Line2D(
                [ax1.get_position().x1, ax2.get_position().x0],
                [1 - (py - 0.5) / (n + 0.5), 1 - (ay - 0.5) / (n + 0.5)],
                transform=fig.transFigure,
                color=c, alpha=0.45, linewidth=1.2,
            )
        )

    fig.suptitle(f"{year} {race_name} GP — predicted vs actual",
                 fontsize=14, y=0.98, color="#CCCCCC")
    fig.tight_layout()
    if save:
        path = _save(fig, f"overlay_{race_name.lower().replace(' ', '_')}_{year}.png")
        return fig, path
    return fig


# ── 5. SHAP waterfall ─────────────────────────────────────────────────────

def shap_waterfall(
    shap_values: pd.Series,
    driver: str,
    race_name: str,
    year: int = 2026,
    top_n: int = 12,
    save: bool = True,
) -> plt.Figure:
    """
    Waterfall chart of SHAP values for one driver.
    Shows which features pushed the prediction up or down.
    Best for explainability posts: 'Here's WHY the model liked VER at Monaco'
    """
    _set_style()

    vals = shap_values.sort_values(key=abs, ascending=False).head(top_n)
    vals = vals.sort_values(ascending=True)

    colors = ["#E8002D" if v < 0 else "#27F4D2" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(vals.index, vals.values, color=colors, height=0.65, alpha=0.88)

    for bar, val in zip(bars, vals.values):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color="#CCCCCC", fontsize=9,
        )

    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to rank score")
    ax.set_title(f"{year} {race_name} GP — {driver}: feature contributions", pad=12)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    pos_patch = mpatches.Patch(color="#27F4D2", label="Increases predicted rank")
    neg_patch = mpatches.Patch(color="#E8002D", label="Decreases predicted rank")
    ax.legend(handles=[pos_patch, neg_patch], framealpha=0.2)

    fig.tight_layout()
    if save:
        path = _save(fig, f"shap_{driver}_{race_name.lower().replace(' ', '_')}_{year}.png")
        return fig, path
    return fig


# ── 6. Feature importance bar ─────────────────────────────────────────────

def feature_importance_bar(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    """
    Global XGBoost feature importance (gain), top N features.
    """
    _set_style()

    df = importance_df.head(top_n).copy()
    df = df.sort_values("Importance", ascending=True)

    # Colour by feature group
    group_colors = {
        "Gap":        "#3671C6",
        "Rolling":    "#FF8000",
        "Form":       "#FF8000",
        "Season":     "#FF8000",
        "Discounted": "#E8002D",
        "Raw":        "#E8002D",
        "Reg":        "#E8002D",
        "PU":         "#E8002D",
        "Circuit":    "#27F4D2",
        "Overtake":   "#27F4D2",
        "Tyre":       "#27F4D2",
        "SC":         "#27F4D2",
        "Weather":    "#229971",
        "Rain":       "#229971",
        "Adaptation": "#FF87BC",
    }

    def feat_color(name):
        for key, col in group_colors.items():
            if key.lower() in name.lower():
                return col
        return "#888888"

    colors = [feat_color(f) for f in df["Feature"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(df["Feature"], df["Importance"], color=colors, height=0.65, alpha=0.88)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + bar.get_width() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{row.get('ImportancePct', 0):.1f}%",
            va="center", color="#CCCCCC", fontsize=8,
        )

    ax.set_xlabel("Feature importance (XGBoost gain)")
    ax.set_title("Global feature importance — F1 2026 predictor", pad=12)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if save:
        path = _save(fig, "feature_importance.png")
        return fig, path
    return fig


# ── 7. Championship trajectory ────────────────────────────────────────────

def championship_trajectory(
    standings_df: pd.DataFrame,
    wdc_probs: pd.DataFrame,
    year: int = 2026,
    top_n: int = 6,
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel figure:
      Top: cumulative points per driver over the season
      Bottom: WDC win probability per driver after each round
    """
    _set_style()

    top_drivers = (
        standings_df.groupby("Driver")["Points"]
        .sum().sort_values(ascending=False)
        .head(top_n).index.tolist()
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    rounds = sorted(standings_df["Round"].unique())

    # Panel 1: cumulative points
    for driver in top_drivers:
        drv_data = standings_df[standings_df["Driver"] == driver].sort_values("Round")
        cum_pts  = drv_data.groupby("Round")["Points"].sum().cumsum().reindex(rounds).ffill()
        ax1.plot(rounds, cum_pts, color=_driver_color(driver),
                 linewidth=2.2, marker="o", markersize=4, label=driver)

    ax1.set_ylabel("Cumulative points")
    ax1.set_title(f"{year} F1 championship — points trajectory", pad=10)
    ax1.legend(framealpha=0.2)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Panel 2: WDC probability
    if not wdc_probs.empty and "Round" in wdc_probs.columns:
        for driver in top_drivers:
            drv_probs = wdc_probs[wdc_probs["Driver"] == driver].sort_values("Round")
            if drv_probs.empty:
                continue
            ax2.plot(
                drv_probs["Round"], drv_probs["WDC_Prob"] * 100,
                color=_driver_color(driver),
                linewidth=2.2, marker="o", markersize=4, label=driver,
            )

        ax2.set_ylabel("WDC win probability (%)")
        ax2.set_xlabel("Race round")
        ax2.set_title(f"{year} WDC probability evolution", pad=10)
        ax2.legend(framealpha=0.2)
        ax2.yaxis.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
        ax2.set_ylim(0, 100)

    fig.tight_layout()
    if save:
        path = _save(fig, f"championship_trajectory_{year}.png")
        return fig, path
    return fig


# ── 8. Circuit DNA radar ───────────────────────────────────────────────────

def circuit_dna_radar(
    circuit_name: str,
    circuit_features: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """
    Spider/radar chart of circuit DNA — great for pre-race content.
    Shows what kind of circuit it is across 5 dimensions.
    """
    _set_style()

    dims   = ["Overtake\nindex", "Tyre deg", "SC\nprob", "Track\nevolution", "Street\ncircuit"]
    values = [
        float(circuit_features.get("OvertakeIndex",  0.5)),
        float(circuit_features.get("TyreDegIndex",   0.05)) * 10,
        float(circuit_features.get("SC_Probability", 0.4)),
        float(circuit_features.get("TrackEvolution", 0.4)),
        float(circuit_features.get("IsStreetCircuit",0)),
    ]
    values_norm = np.clip(values, 0, 1)

    n      = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values_plot = list(values_norm) + [values_norm[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_facecolor("#0F0F0F")

    ax.plot(angles, values_plot, color="#FF8000", linewidth=2, linestyle="solid")
    ax.fill(angles, values_plot, alpha=0.25, color="#FF8000")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10, color="#CCCCCC")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="#666666")
    ax.set_ylim(0, 1)
    ax.grid(color="#333333", linewidth=0.5)
    ax.spines["polar"].set_edgecolor("#333333")
    ax.set_title(f"{circuit_name} — circuit DNA", pad=20, color="#CCCCCC", fontsize=13)

    fig.tight_layout()
    if save:
        path = _save(fig, f"circuit_dna_{circuit_name.lower().replace(' ', '_')}.png")
        return fig, path
    return fig


# ── 9. Model accuracy tracker ─────────────────────────────────────────────

def model_accuracy_tracker(
    accuracy_log: pd.DataFrame,
    year: int = 2026,
    save: bool = True,
) -> plt.Figure:
    """
    Four-panel accuracy dashboard across the season.
    Auto-updates after each race — the living scoreboard on the README.
    """
    _set_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rounds = accuracy_log["Round"].values if "Round" in accuracy_log.columns else range(len(accuracy_log))

    metrics = [
        ("win_correct",     "Win accuracy",          "#27F4D2", 0, 1),
        ("spearman_rho",    "Spearman rank corr (ρ)", "#FF8000", -1, 1),
        ("mae_positions",   "MAE (positions)",        "#E8002D", 0, 10),
        ("brier_win",       "Brier score (win)",      "#FF87BC", 0, 1),
    ]

    for ax, (col, label, color, ymin, ymax) in zip(axes.flat, metrics):
        if col not in accuracy_log.columns:
            ax.text(0.5, 0.5, f"{label}\n(no data yet)", ha="center", va="center",
                    transform=ax.transAxes, color="#666666")
            ax.set_title(label, fontsize=11)
            continue

        vals     = accuracy_log[col].values
        roll_avg = pd.Series(vals).rolling(3, min_periods=1).mean().values

        ax.scatter(rounds, vals, color=color, alpha=0.7, s=40, zorder=3)
        ax.plot(rounds, roll_avg, color=color, linewidth=2, alpha=0.9,
                label="3-race avg")

        # Reference line (random baseline or target)
        if col == "win_correct":
            ax.axhline(0.30, color="#666666", linestyle="--", linewidth=0.8,
                       label="Target (30%)")
        elif col == "spearman_rho":
            ax.axhline(0.60, color="#666666", linestyle="--", linewidth=0.8,
                       label="Target (0.60)")
        elif col == "brier_win":
            ax.axhline(1/20, color="#666666", linestyle="--", linewidth=0.8,
                       label=f"Random ({1/20:.2f})")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Race round", fontsize=9)
        ax.set_ylim(ymin - 0.05, ymax + 0.05)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(framealpha=0.2, fontsize=8)

    fig.suptitle(f"{year} season — model accuracy tracker", fontsize=14, y=1.01)
    fig.tight_layout()
    if save:
        path = _save(fig, f"accuracy_tracker_{year}.png")
        return fig, path
    return fig


# ── Batch generator ───────────────────────────────────────────────────────

def generate_race_week_plots(
    prediction: pd.DataFrame,
    race_name: str,
    year: int = 2026,
    actual: pd.DataFrame = None,
    shap_df: pd.DataFrame = None,
    circuit_features: pd.Series = None,
) -> dict:
    """
    Generate the full set of race-week charts in one call.
    Returns dict of {chart_name: file_path}.
    """
    import logging; logger = logging.getLogger(__name__)
    outputs = {}

    try:
        _, p = win_probability_chart(prediction, race_name, year)
        outputs["win_prob"] = p
        logger.info(f"  win_probability_chart → {p.name}")
    except Exception as e:
        logger.warning(f"  win_prob failed: {e}")

    try:
        _, p = podium_heatmap(prediction, race_name, year)
        outputs["podium_heatmap"] = p
        logger.info(f"  podium_heatmap → {p.name}")
    except Exception as e:
        logger.warning(f"  podium_heatmap failed: {e}")

    try:
        _, p = position_distribution(prediction, race_name, year)
        outputs["pos_dist"] = p
        logger.info(f"  position_distribution → {p.name}")
    except Exception as e:
        logger.warning(f"  pos_dist failed: {e}")

    if actual is not None and not actual.empty:
        try:
            _, p = pre_post_race_overlay(prediction, actual, race_name, year)
            outputs["overlay"] = p
            logger.info(f"  pre_post_overlay → {p.name}")
        except Exception as e:
            logger.warning(f"  overlay failed: {e}")

    if shap_df is not None and not shap_df.empty:
        top_driver = prediction.iloc[0]["Driver"]
        try:
            _, p = shap_waterfall(shap_df.loc[top_driver], top_driver, race_name, year)
            outputs["shap"] = p
            logger.info(f"  shap_waterfall ({top_driver}) → {p.name}")
        except Exception as e:
            logger.warning(f"  shap failed: {e}")

    if circuit_features is not None:
        try:
            _, p = circuit_dna_radar(race_name, circuit_features)
            outputs["circuit_dna"] = p
            logger.info(f"  circuit_dna_radar → {p.name}")
        except Exception as e:
            logger.warning(f"  circuit_dna failed: {e}")

    return outputs


if __name__ == "__main__":
    print("Testing visualisations with synthetic data...")

    np.random.seed(42)
    drivers = ["VER","NOR","LEC","HAM","RUS","PIA","SAI","ALO",
               "STR","GAS","ALB","TSU","OCO","BEA","HUL","BOR","DOO","HAD","ANT","LAW"]
    teams   = [DRIVER_TEAM.get(d, "Unknown") for d in drivers]

    win_probs = np.array([0.27,0.25,0.18,0.13,0.06,0.03,0.02,0.01,
                          0.01,0.01,0.01,0.005,0.005,0.003,0.003,0.002,0.002,0.002,0.001,0.001])
    win_probs /= win_probs.sum()

    prediction = pd.DataFrame({
        "Driver":         drivers,
        "Team":           teams,
        "PredictedPos":   range(1, 21),
        "WinProb":        win_probs,
        "P2Prob":         win_probs * 0.8,
        "P3Prob":         win_probs * 0.7,
        "PodiumProb":     win_probs * 2.2,
        "Top10Prob":      np.clip(win_probs * 8, 0, 1),
        "DNFProb":        np.random.uniform(0.04, 0.15, 20),
        "ExpectedPos":    range(1, 21),
        "PosStdDev":      np.random.uniform(1.5, 4.5, 20),
        "WinProb_CI_low": win_probs * 0.85,
        "WinProb_CI_high":win_probs * 1.15,
    })
    for i in range(1, 11):
        base = np.clip(win_probs * (1.5 - i * 0.1), 0, 1)
        prediction[f"P{i}_prob"] = base / base.sum() * (1 / i)

    race_name = "Bahrain"

    plots = generate_race_week_plots(
        prediction = prediction,
        race_name  = race_name,
        year       = 2026,
    )

    print(f"\nGenerated {len(plots)} plots:")
    for name, path in plots.items():
        print(f"  {name}: {path}")
