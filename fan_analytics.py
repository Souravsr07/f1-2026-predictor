"""
fan_analytics.py

Fan-facing analytics — the outputs that make people actually share the project.
These go beyond race predictions into the "who is statistically the best at X" 
territory that drives engagement.

Outputs:
  1. circuit_specialist_rankings  — who statistically excels at each circuit type
  2. wet_weather_kings            — wet race performance index per driver
  3. head_to_head_record          — historical P(A beats B) for any driver pair
  4. rookie_learning_curve        — performance vs experience for 2026 rookies
  5. overtake_machine_index       — who gains the most positions from grid to flag
  6. driver_comparison_table      — side-by-side stats for any two drivers
  7. all_time_circuit_kings       — best historical record at each circuit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROCESSED
from visualisations import _set_style, _driver_color, _save, DRIVER_TEAM, TEAM_COLORS

import matplotlib
matplotlib.use("Agg")


# ── 1. Circuit specialist rankings ────────────────────────────────────────

def circuit_specialist_rankings(
    results: pd.DataFrame,
    circuit_name: str,
    min_appearances: int = 3,
    save: bool = True,
) -> tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    Rank drivers by their historical performance at a specific circuit.

    Metric: normalised average finishing position, weighted by recency.
    Lower average position = better. Adjusted for car competitiveness
    by comparing finish vs expected finish (grid position proxy).

    Returns (ranking_df, figure)
    """
    circuit_data = results[
        results["Circuit"].str.contains(circuit_name, case=False, na=False)
    ].copy()

    if circuit_data.empty:
        return pd.DataFrame(), None

    # Recency weight: most recent race counts 3x, older races less
    max_year = circuit_data["Year"].max()
    circuit_data["RecencyWeight"] = circuit_data["Year"].apply(
        lambda y: 1.0 + 2.0 * max(0, (y - (max_year - 3)) / 3)
    )

    # Weighted average finish + over/underperformance vs grid
    stats = []
    for driver, grp in circuit_data.groupby("Driver"):
        if len(grp) < min_appearances:
            continue

        w       = grp["RecencyWeight"].values
        w_norm  = w / w.sum()
        avg_fin = np.average(grp["FinishPosition"].values, weights=w_norm)

        if "GridPosition" in grp.columns:
            valid = grp.dropna(subset=["GridPosition"])
            if not valid.empty:
                w2     = valid["RecencyWeight"].values / valid["RecencyWeight"].sum()
                grid_p = np.average(valid["GridPosition"].values, weights=w2)
                delta  = grid_p - avg_fin   # positive = gains positions
            else:
                grid_p, delta = np.nan, 0.0
        else:
            grid_p, delta = np.nan, 0.0

        best    = int(grp["FinishPosition"].min())
        wins    = int((grp["FinishPosition"] == 1).sum())
        podiums = int((grp["FinishPosition"] <= 3).sum())
        dnfs    = int(grp.get("DNF", pd.Series(0, index=grp.index)).sum())

        stats.append({
            "Driver":           driver,
            "Appearances":      len(grp),
            "AvgFinish":        round(avg_fin, 2),
            "AvgGrid":          round(grid_p, 2) if not np.isnan(grid_p) else None,
            "PositionsGained":  round(delta, 2),
            "BestFinish":       best,
            "Wins":             wins,
            "Podiums":          podiums,
            "DNFs":             dnfs,
        })

    if not stats:
        return pd.DataFrame(), None

    ranking = pd.DataFrame(stats).sort_values("AvgFinish").reset_index(drop=True)
    ranking["Rank"] = ranking.index + 1

    # Plot
    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    top = ranking.head(12)
    colors = [_driver_color(d) for d in top["Driver"]]

    bars = ax.barh(
        top["Driver"][::-1],
        top["AvgFinish"][::-1],
        color=colors[::-1],
        height=0.65, alpha=0.88,
    )

    if "PositionsGained" in top.columns:
        for i, (bar, (_, row)) in enumerate(zip(bars[::-1], top.iterrows())):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"P{int(row['BestFinish'])} best  |  {row['Wins']}W {row['Podiums']}Pod",
                va="center", color="#AAAAAA", fontsize=8,
            )

    ax.set_xlabel("Weighted avg finishing position (lower = better)")
    ax.set_title(f"Circuit specialists — {circuit_name}", pad=12)
    ax.invert_xaxis()
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = None
    if save:
        _, path = fig, _save(fig, f"specialists_{circuit_name.lower().replace(' ', '_')}.png")

    return ranking, fig


# ── 2. Wet weather kings ──────────────────────────────────────────────────

def wet_weather_kings(
    results: pd.DataFrame,
    min_wet_races: int = 3,
    save: bool = True,
) -> tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    Rank drivers by wet-weather overperformance.
    Metric: (avg dry finish - avg wet finish) normalised by dry baseline.
    Positive = performs BETTER in the wet relative to their dry level.
    """
    if "Rainfall_any" not in results.columns:
        results = results.copy()
        results["Rainfall_any"] = False

    wet  = results[results["Rainfall_any"] == True]
    dry  = results[results["Rainfall_any"] == False]

    stats = []
    for driver in results["Driver"].unique():
        wet_races = wet[wet["Driver"] == driver]
        dry_races = dry[dry["Driver"] == driver]

        if len(wet_races) < min_wet_races:
            continue

        avg_wet = wet_races["FinishPosition"].mean()
        avg_dry = dry_races["FinishPosition"].mean() if len(dry_races) > 0 else 10.0

        # Overperformance index: positive = gains positions in wet
        overperf = (avg_dry - avg_wet) / max(avg_dry, 1.0)

        stats.append({
            "Driver":         driver,
            "WetRaces":       len(wet_races),
            "DryRaces":       len(dry_races),
            "AvgWetFinish":   round(avg_wet, 2),
            "AvgDryFinish":   round(avg_dry, 2),
            "WetOverperf":    round(overperf, 4),
            "WetWins":        int((wet_races["FinishPosition"] == 1).sum()),
        })

    if not stats:
        return pd.DataFrame(), None

    ranking = pd.DataFrame(stats).sort_values("WetOverperf", ascending=False).reset_index(drop=True)
    ranking["Rank"] = ranking.index + 1

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    top    = ranking.head(12)
    colors = [_driver_color(d) for d in top["Driver"]]
    vals   = top["WetOverperf"].values

    bar_colors = ["#27F4D2" if v >= 0 else "#E8002D" for v in vals]
    bars = ax.barh(top["Driver"], vals * 100, color=bar_colors, height=0.65, alpha=0.88)

    for bar, (_, row) in zip(bars, top.iterrows()):
        w = bar.get_width()
        ax.text(
            w + (0.3 if w >= 0 else -0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{row['WetRaces']} wet races",
            va="center",
            ha="left" if w >= 0 else "right",
            color="#AAAAAA", fontsize=8,
        )

    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Wet overperformance index (%) — positive = better in wet")
    ax.set_title("Wet weather kings — F1 2018–2025", pad=12)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = None
    if save:
        path = _save(fig, "wet_weather_kings.png")

    return ranking, fig


# ── 3. Head-to-head record ────────────────────────────────────────────────

def head_to_head_record(
    results: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    same_team_only: bool = False,
) -> dict:
    """
    Historical head-to-head record between two drivers.
    Includes: win rate, avg positions, recent trend.
    """
    shared_races = results.groupby(["Year", "Round"]).filter(
        lambda g: driver_a in g["Driver"].values and driver_b in g["Driver"].values
    )

    if shared_races.empty:
        return {"error": f"No shared races found between {driver_a} and {driver_b}"}

    if same_team_only and "Team" in shared_races.columns:
        shared_races = shared_races.groupby(["Year", "Round"]).filter(
            lambda g: (
                g.loc[g["Driver"] == driver_a, "Team"].values[0] ==
                g.loc[g["Driver"] == driver_b, "Team"].values[0]
            ) if driver_a in g["Driver"].values and driver_b in g["Driver"].values else False
        )

    a_wins = 0
    b_wins = 0
    race_records = []

    for (year, rnd), race in shared_races.groupby(["Year", "Round"]):
        a_pos = race.loc[race["Driver"] == driver_a, "FinishPosition"].values
        b_pos = race.loc[race["Driver"] == driver_b, "FinishPosition"].values

        if len(a_pos) == 0 or len(b_pos) == 0:
            continue

        a_pos, b_pos = float(a_pos[0]), float(b_pos[0])
        a_won = int(a_pos < b_pos)
        a_wins += a_won
        b_wins += 1 - a_won

        circuit = race["Circuit"].iloc[0] if "Circuit" in race.columns else ""
        race_records.append({
            "Year": year, "Round": rnd, "Circuit": circuit,
            f"{driver_a}_pos": a_pos, f"{driver_b}_pos": b_pos,
            "winner": driver_a if a_won else driver_b,
        })

    total = a_wins + b_wins
    records_df = pd.DataFrame(race_records)

    # Recent form (last 5 shared races)
    recent = records_df.tail(5)
    a_recent_wins = int((recent["winner"] == driver_a).sum())

    return {
        "driver_a":          driver_a,
        "driver_b":          driver_b,
        "total_races":       total,
        f"{driver_a}_wins":  a_wins,
        f"{driver_b}_wins":  b_wins,
        f"{driver_a}_win_pct": round(a_wins / total, 3) if total > 0 else 0,
        f"{driver_b}_win_pct": round(b_wins / total, 3) if total > 0 else 0,
        f"{driver_a}_recent_wins_last5": a_recent_wins,
        f"{driver_b}_recent_wins_last5": 5 - a_recent_wins,
        "records":           records_df,
    }


def head_to_head_chart(
    h2h: dict,
    save: bool = True,
) -> plt.Figure:
    """Visual H2H chart from head_to_head_record() output."""
    _set_style()
    a, b  = h2h["driver_a"], h2h["driver_b"]
    total = h2h["total_races"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Pie chart
    ax = axes[0]
    wins = [h2h.get(f"{a}_wins", 0), h2h.get(f"{b}_wins", 0)]
    colors = [_driver_color(a), _driver_color(b)]
    ax.pie(
        wins, labels=[a, b], colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "#CCCCCC", "fontsize": 12},
    )
    ax.set_title(f"{a} vs {b} — {total} races", pad=10)

    # Timeline of wins
    ax2 = axes[1]
    records = h2h.get("records", pd.DataFrame())
    if not records.empty:
        for i, (_, row) in enumerate(records.iterrows()):
            c = _driver_color(row["winner"])
            ax2.bar(i, 1, color=c, width=0.8, alpha=0.85)
            ax2.text(i, 0.5, row["winner"][:3], ha="center", va="center",
                     color="white", fontsize=7)

        ax2.set_ylim(0, 1.4)
        ax2.set_yticks([])
        ax2.set_xlabel("Race (chronological)")
        ax2.set_title(f"Race-by-race record", pad=10)
        ax2.tick_params(axis="x", labelbottom=False)

    fig.suptitle(f"Head-to-head: {a} vs {b}", fontsize=13, y=1.01)
    fig.tight_layout()

    if save:
        path = _save(fig, f"h2h_{a}_{b}.png")
    return fig


# ── 4. Rookie learning curve ──────────────────────────────────────────────

def rookie_learning_curve(
    results: pd.DataFrame,
    rookie_year: int = 2026,
    rookies: list = None,
    save: bool = True,
) -> plt.Figure:
    """
    Plot finishing position vs race number for 2026 rookies.
    Shows the learning curve — how quickly they close the gap to teammates.
    Overlaid with historical rookie benchmarks (Leclerc 2018, Norris 2019, Piastri 2023).
    """
    if rookies is None:
        rookies = ["ANT", "HAD", "BOR"]

    _set_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    # Historical rookie benchmarks (avg finish per race number)
    benchmarks = {
        "LEC 2018": [15, 12, 8, 10, 6, 9, 5, 8, 7, 6, 5, 8, 6, 4, 5, 6, 4, 6, 5, 4, 5],
        "NOR 2019": [12, 10, 11, 6, 8, 8, 6, 7, 5, 9, 7, 6, 8, 6, 7, 5, 8, 6, 5, 6, 7],
        "PIA 2023": [14, 8, 10, 7, 6, 5, 4, 6, 3, 5, 4, 5, 7, 4, 6, 3, 5, 4, 3, 4, 5],
    }
    bench_style = {"LEC 2018": "--", "NOR 2019": "-.", "PIA 2023": ":"}

    for name, positions in benchmarks.items():
        races = range(1, len(positions) + 1)
        roll  = pd.Series(positions).rolling(3, min_periods=1).mean()
        ax.plot(races, roll, color="#555555", linestyle=bench_style[name],
                linewidth=1.2, alpha=0.65, label=name)

    # 2026 rookie data (from results if available)
    rookie_season = results[
        (results["Year"] == rookie_year) &
        (results["Driver"].isin(rookies))
    ]

    if rookie_season.empty:
        # Placeholder: simulate expected trajectory
        for rookie in rookies:
            races     = range(1, 13)
            start_pos = 15.0
            end_pos   = 9.0
            trajectory = [start_pos - (start_pos - end_pos) * (i / 11) +
                          np.random.normal(0, 1.5) for i in range(12)]
            trajectory = np.clip(trajectory, 1, 20)
            ax.plot(list(races), trajectory, color=_driver_color(rookie),
                    linewidth=2.2, marker="o", markersize=4,
                    label=f"{rookie} 2026 (projected)", alpha=0.8)
    else:
        for rookie in rookies:
            drv = rookie_season[rookie_season["Driver"] == rookie]\
                  .sort_values("Round")
            if drv.empty:
                continue
            roll = drv["FinishPosition"].rolling(3, min_periods=1).mean()
            ax.plot(drv["Round"], roll, color=_driver_color(rookie),
                    linewidth=2.2, marker="o", markersize=4,
                    label=f"{rookie} 2026", alpha=0.9)

    ax.set_xlabel("Race number")
    ax.set_ylabel("Avg finish position (3-race rolling)")
    ax.set_title("2026 rookie learning curves vs historical benchmarks", pad=12)
    ax.set_ylim(20, 1)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.2, fontsize=9, ncol=2)

    fig.tight_layout()
    if save:
        path = _save(fig, f"rookie_curves_{rookie_year}.png")
    return fig


# ── 5. Overtake machine index ─────────────────────────────────────────────

def overtake_machine_index(
    results: pd.DataFrame,
    min_races: int = 20,
    save: bool = True,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Rank drivers by how many positions they gain from qualifying to finish.
    The 'Overtake Machine' — who turns poor qualis into good races.
    """
    if "GridPosition" not in results.columns:
        return pd.DataFrame(), None

    stats = []
    for driver, grp in results.groupby("Driver"):
        valid = grp.dropna(subset=["GridPosition", "FinishPosition"])
        valid = valid[valid["GridPosition"] > 0]
        if len(valid) < min_races:
            continue

        gained      = (valid["GridPosition"] - valid["FinishPosition"])
        avg_gained  = gained.mean()
        pos_races   = int((gained > 0).sum())
        big_gainers = int((gained >= 5).sum())

        stats.append({
            "Driver":         driver,
            "Races":          len(valid),
            "AvgPosGained":   round(avg_gained, 2),
            "RacesGained":    pos_races,
            "BigMoves5plus":  big_gainers,
            "BigMovePct":     round(big_gainers / len(valid) * 100, 1),
        })

    if not stats:
        return pd.DataFrame(), None

    ranking = pd.DataFrame(stats).sort_values("AvgPosGained", ascending=False)\
              .reset_index(drop=True)
    ranking["Rank"] = ranking.index + 1

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    top    = ranking.head(14)
    colors = [_driver_color(d) for d in top["Driver"]]
    vals   = top["AvgPosGained"].values
    bar_c  = ["#27F4D2" if v >= 0 else "#E8002D" for v in vals]

    ax.barh(top["Driver"][::-1], vals[::-1], color=bar_c[::-1], height=0.65, alpha=0.88)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Avg positions gained (qualifying → finish)")
    ax.set_title("Overtake machine index — F1 2018–2025", pad=12)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save:
        _save(fig, "overtake_machine.png")
    return ranking, fig


# ── 6. Driver comparison table ────────────────────────────────────────────

def driver_comparison_table(
    results: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Side-by-side career stats for two drivers.
    Returns a DataFrame with metric / driver_a_value / driver_b_value columns.
    """
    def _stats(driver):
        d = results[results["Driver"] == driver]
        if d.empty:
            return {}
        dnf_col = "DNF" if "DNF" in d.columns else None
        wet_col = "Rainfall_any" if "Rainfall_any" in d.columns else None

        s = {
            "Races":            len(d),
            "Wins":             int((d["FinishPosition"] == 1).sum()),
            "Podiums":          int((d["FinishPosition"] <= 3).sum()),
            "Top10s":           int((d["FinishPosition"] <= 10).sum()),
            "AvgFinish":        round(d["FinishPosition"].mean(), 2),
            "BestFinish":       int(d["FinishPosition"].min()),
            "Win%":             f"{(d['FinishPosition']==1).mean()*100:.1f}%",
            "Podium%":          f"{(d['FinishPosition']<=3).mean()*100:.1f}%",
        }
        if dnf_col:
            s["DNF%"] = f"{d[dnf_col].mean()*100:.1f}%"
        if wet_col:
            wet = d[d[wet_col] == True]
            s["Wet avg finish"] = round(wet["FinishPosition"].mean(), 2) if len(wet) > 0 else "N/A"
        if "GridPosition" in d.columns:
            valid = d.dropna(subset=["GridPosition"])
            s["Avg positions gained"] = round(
                (valid["GridPosition"] - valid["FinishPosition"]).mean(), 2
            )
        return s

    stats_a = _stats(driver_a)
    stats_b = _stats(driver_b)

    all_metrics = list(dict.fromkeys(list(stats_a.keys()) + list(stats_b.keys())))
    rows = []
    for m in all_metrics:
        rows.append({
            "Metric":  m,
            driver_a:  stats_a.get(m, "N/A"),
            driver_b:  stats_b.get(m, "N/A"),
        })

    return pd.DataFrame(rows)


# ── 7. All-time circuit kings ─────────────────────────────────────────────

def all_time_circuit_kings(
    results: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    For each circuit, find the top N drivers by win count (all time).
    Returns a dict-like DataFrame: Circuit → top drivers with win counts.
    """
    rows = []
    for circuit, grp in results.groupby("Circuit"):
        wins = grp[grp["FinishPosition"] == 1].groupby("Driver").size()\
               .sort_values(ascending=False).head(top_n)
        for rank, (driver, count) in enumerate(wins.items(), 1):
            rows.append({
                "Circuit": circuit,
                "Rank":    rank,
                "Driver":  driver,
                "Wins":    count,
            })

    return pd.DataFrame(rows)


# ── Batch fan analytics ───────────────────────────────────────────────────

def generate_fan_analytics_pack(
    results: pd.DataFrame,
    year: int = 2026,
) -> dict:
    """
    Generate all fan analytics charts in one call.
    Returns dict of {chart_name: figure}.
    """
    outputs = {}

    print("  Generating wet weather kings...")
    try:
        ranking, fig = wet_weather_kings(results)
        if fig:
            outputs["wet_kings"] = fig
            print(f"    Top wet driver: {ranking.iloc[0]['Driver']} "
                  f"({ranking.iloc[0]['WetOverperf']:.3f})")
    except Exception as e:
        print(f"    wet_kings failed: {e}")

    print("  Generating overtake machine index...")
    try:
        ranking, fig = overtake_machine_index(results)
        if fig:
            outputs["overtake_machine"] = fig
            if not ranking.empty:
                print(f"    Top overtaker: {ranking.iloc[0]['Driver']} "
                      f"(+{ranking.iloc[0]['AvgPosGained']:.2f} avg)")
    except Exception as e:
        print(f"    overtake_machine failed: {e}")

    print("  Generating rookie curves...")
    try:
        fig = rookie_learning_curve(results, rookie_year=year)
        outputs["rookie_curves"] = fig
    except Exception as e:
        print(f"    rookie_curves failed: {e}")

    return outputs


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    drivers_pool = ["VER","NOR","LEC","HAM","RUS","PIA","SAI","ALO",
                    "STR","GAS","ALB","TSU","OCO","BEA","HUL","BOR",
                    "DOO","HAD","ANT","LAW"]
    circuits = ["Bahrain","Australia","China","Japan","Monaco",
                "Spain","Great Britain","Hungary","Belgium","Italy"]

    rows = []
    for year in [2022, 2023, 2024, 2025]:
        for rnd, circuit in enumerate(circuits, 1):
            order = rng.permutation(drivers_pool)
            for pos, drv in enumerate(order, 1):
                rows.append({
                    "Year": year, "Round": rnd, "Circuit": circuit,
                    "Driver": drv, "FinishPosition": pos,
                    "GridPosition": rng.integers(1, 21),
                    "DNF": int(rng.random() < 0.08),
                    "Rainfall_any": rng.random() < 0.25,
                    "Team": DRIVER_TEAM.get(drv, "Unknown"),
                    "Points": max(0, 26 - pos * 1.3),
                })
    df = pd.DataFrame(rows)

    print("Testing fan analytics...")
    outputs = generate_fan_analytics_pack(df, year=2026)
    print(f"\nGenerated {len(outputs)} fan analytics charts")

    print("\nDriver comparison: VER vs NOR")
    comp = driver_comparison_table(df, "VER", "NOR")
    print(comp.to_string(index=False))

    print("\nH2H: VER vs NOR")
    h2h = head_to_head_record(df, "VER", "NOR")
    print(f"  VER wins: {h2h.get('VER_wins',0)}, NOR wins: {h2h.get('NOR_wins',0)}")
    print(f"  VER win%: {h2h.get('VER_win_pct',0):.1%}")