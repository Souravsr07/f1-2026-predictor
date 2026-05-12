from __future__ import annotations

import argparse
import html
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


TEAM_COLORS = {
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Red Bull": "#3671C6",
    "Racing Bulls": "#6692FF",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "Haas": "#B6BABD",
    "Audi": "#52E252",
    "Cadillac": "#9B0000",
    "Kick Sauber": "#00E701",
}

MODEL_LABELS = {
    "ensemble": "Ensemble stack",
    "quali_baseline": "Qualifying baseline",
    "season_points_baseline": "Season-points baseline",
}

WATCHLIST_ITEMS = [
    (
        "Circuit DNA join mismatch",
        "Historical race names and circuit archetype labels do not currently line up cleanly, so circuit-specific features are muted in the saved matrix.",
    ),
    (
        "Wet-weather signal is flat",
        "Saved feature artifacts currently show no live variation in wet-race fields, which means rain sensitivity is materially under-modelled.",
    ),
    (
        "Constructor priors need year awareness",
        "Historical training rows are still leaning on 2025-style constructor anchors instead of season-specific constructor state, which compresses regime-change learning.",
    ),
]


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_num(value: float, digits: int = 0) -> str:
    if digits == 0:
        return f"{int(round(value)):,}"
    return f"{value:,.{digits}f}"


def fmt_date(value: datetime) -> str:
    return value.strftime("%B %d, %Y").replace(" 0", " ")


def gp_name(name: str) -> str:
    return name if "Grand Prix" in name else f"{name} Grand Prix"


def team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#9AA4B2")


def detect_source_root(cli_value: str | None) -> Path:
    candidates = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[1],
            Path.home() / "Documents" / "f1-2026-predictor",
            Path.home() / "OneDrive" / "Documents" / "New project" / "f1-2026-predictor",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        if (
            (candidate / "results" / "wdc_forecast_2026.csv").exists()
            and (candidate / "data" / "processed" / "2026_live_results.parquet").exists()
        ):
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a project root with live outputs. Checked:\n"
        f"{searched}"
    )


def slugify(value: str) -> str:
    allowed = []
    for ch in value.lower():
        if ch.isalnum():
            allowed.append(ch)
        elif ch in {" ", "-", "_"}:
            allowed.append("_")
    return "".join(allowed).strip("_") or "prediction"


def latest_prediction_paths(source_root: Path) -> tuple[Path, Path]:
    results_dir = source_root / "results"
    return (
        results_dir / "latest_race_prediction.csv",
        results_dir / "latest_race_prediction_meta.json",
    )


def latest_dashboard_path(source_root: Path) -> Path:
    return source_root / "dashboard" / "f1_2026_portfolio_dashboard.html"


def repo_relative_path(path: Path, source_root: Path) -> str:
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def dashboard_history_dir(source_root: Path) -> Path:
    return source_root / "dashboard" / "history"


def archive_dashboard_filename(
    race_name: str,
    year: int,
    race_number: int | None = None,
    generated_at: str | None = None,
) -> str:
    if generated_at:
        try:
            stamp = datetime.fromisoformat(generated_at).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            stamp = generated_at.replace(":", "").replace("-", "").replace("T", "_")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    round_label = f"r{int(race_number)}_" if race_number is not None else ""
    return f"dashboard_{year}_{round_label}{slugify(race_name)}_{stamp}.html"


def save_latest_prediction_artifacts(
    prediction: pd.DataFrame,
    race_name: str,
    year: int,
    race_number: int | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Path]:
    resolved_root = detect_source_root(str(source_root)) if source_root else detect_source_root(None)
    results_dir = resolved_root / "results"
    archive_dir = results_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    latest_csv, latest_meta = latest_prediction_paths(resolved_root)
    generated_at = datetime.now().isoformat(timespec="seconds")
    stamp = datetime.fromisoformat(generated_at).strftime("%Y%m%d_%H%M%S")
    round_label = f"r{int(race_number)}_" if race_number is not None else ""
    archive_csv = archive_dir / f"prediction_{year}_{round_label}{slugify(race_name)}_{stamp}.csv"

    prediction.to_csv(latest_csv, index=False)
    prediction.to_csv(archive_csv, index=False)

    metadata = {
        "race_name": race_name,
        "year": int(year),
        "race_number": int(race_number) if race_number is not None else None,
        "generated_at": generated_at,
        "latest_csv": repo_relative_path(latest_csv, resolved_root),
        "archive_csv": repo_relative_path(archive_csv, resolved_root),
    }
    latest_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "latest_csv": latest_csv,
        "latest_meta": latest_meta,
        "archive_csv": archive_csv,
        "generated_at": generated_at,
    }


def save_dashboard_archive(
    race_name: str,
    year: int,
    race_number: int | None = None,
    source_root: str | Path | None = None,
    generated_at: str | None = None,
    html_text: str | None = None,
) -> Path:
    resolved_root = detect_source_root(str(source_root)) if source_root else detect_source_root(None)
    history_dir = dashboard_history_dir(resolved_root)
    history_dir.mkdir(parents=True, exist_ok=True)

    archive_path = history_dir / archive_dashboard_filename(
        race_name=race_name,
        year=year,
        race_number=race_number,
        generated_at=generated_at,
    )

    if html_text is None:
        html_text = latest_dashboard_path(resolved_root).read_text(encoding="utf-8")

    archive_path.write_text(html_text, encoding="utf-8")
    return archive_path


def load_config_module(source_root: Path):
    config_path = source_root / "config.py"
    spec = importlib.util.spec_from_file_location("f1_dashboard_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_eda_inputs(source_root: Path) -> None:
    eda_dir = source_root / "results" / "eda"
    required = [
        eda_dir / "feature_target_correlation.csv",
        eda_dir / "live_2026_team_shift.csv",
    ]
    if all(path.exists() for path in required):
        return

    run_eda_path = source_root / "scripts" / "run_eda.py"
    spec = importlib.util.spec_from_file_location("f1_dashboard_run_eda", run_eda_path)
    if spec is None or spec.loader is None:
        missing = ", ".join(str(path) for path in required if not path.exists())
        raise FileNotFoundError(f"Missing EDA inputs and cannot load {run_eda_path}: {missing}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    still_missing = [path for path in required if not path.exists()]
    if still_missing:
        missing = ", ".join(str(path) for path in still_missing)
        raise FileNotFoundError(f"EDA generation completed but required dashboard inputs are missing: {missing}")


def render_form_tokens(last3: str) -> str:
    tokens = []
    for token in last3.split():
        try:
            position = int(token.replace("P", ""))
        except ValueError:
            position = 99
        if position <= 3:
            css = "good"
        elif position <= 10:
            css = "mid"
        else:
            css = "bad"
        tokens.append(f"<span class='form-pill {css}'>{html.escape(token)}</span>")
    return "".join(tokens)


def render_wdc_rows(wdc: pd.DataFrame) -> str:
    max_points = max(float(wdc["ExpectedFinalPoints"].max()), 1.0)
    rows = []
    for _, row in wdc.head(6).iterrows():
        width = min(100.0, float(row["ExpectedFinalPoints"]) / max_points * 100)
        rows.append(
            f"""
            <div class="prob-row" style="--team-color:{team_color(row['Team'])}">
              <div class="prob-head">
                <div>
                  <div class="driver-line">
                    <span class="team-swatch" style="background:{team_color(row['Team'])}"></span>
                    <span>{html.escape(row['DriverFull'])}</span>
                  </div>
                  <div class="prob-meta">{html.escape(row['Team'])} | {fmt_num(row['CurrentPoints'])} pts now | {fmt_num(row['ExpectedFinalPoints'], 1)} expected</div>
                </div>
                <div class="prob-value">{fmt_pct(row['WDC_Prob'])}</div>
              </div>
              <div class="prob-bar"><span style="width:{width:.1f}%;background:linear-gradient(90deg,{team_color(row['Team'])},rgba(255,255,255,0.18))"></span></div>
              <div class="range-note">P10-P90 final points: {fmt_num(row['P10_Points'])} to {fmt_num(row['P90_Points'])}</div>
            </div>
            """
        )
    return "".join(rows)


def render_driver_table(standings: pd.DataFrame) -> str:
    rows = []
    for _, row in standings.head(10).iterrows():
        rows.append(
            f"""
            <tr>
              <td>{int(row['Position'])}</td>
              <td>
                <div class="driver-cell">
                  <span class="team-swatch" style="background:{team_color(row['Team'])}"></span>
                  <div>
                    <div class="driver-name">{html.escape(row['DriverFull'])}</div>
                    <div class="table-meta">{html.escape(row['Driver'])} | {html.escape(row['Team'])}</div>
                  </div>
                </div>
              </td>
              <td>{fmt_num(row['TotalPoints'])}</td>
              <td>{int(row['Wins'])}</td>
              <td>{int(row['Podiums'])}</td>
              <td>{row['AvgFinish']:.2f}</td>
              <td>{render_form_tokens(row['Last3'])}</td>
            </tr>
            """
        )
    return "".join(rows)


def render_constructor_rows(constructors: pd.DataFrame) -> str:
    leader_points = max(float(constructors["Points"].max()), 1.0)
    rows = []
    for _, row in constructors.head(7).iterrows():
        width = float(row["Points"]) / leader_points * 100
        rows.append(
            f"""
            <div class="team-row" style="--team-color:{team_color(row['Team'])}">
              <div class="team-row-head">
                <div class="team-row-name">
                  <span class="rank-chip">{int(row['Position'])}</span>
                  <span class="team-swatch" style="background:{team_color(row['Team'])}"></span>
                  <span>{html.escape(row['Team'])}</span>
                </div>
                <div class="team-row-value">{fmt_num(row['Points'])} pts</div>
              </div>
              <div class="prob-bar slim"><span style="width:{width:.1f}%;background:linear-gradient(90deg,{team_color(row['Team'])},rgba(255,255,255,0.18))"></span></div>
            </div>
            """
        )
    return "".join(rows)


def render_shift_rows(team_shift: pd.DataFrame) -> str:
    pos = team_shift[team_shift["PointsShift"] > 0].sort_values("PointsShift", ascending=False).head(4)
    neg = team_shift[team_shift["PointsShift"] < 0].sort_values("PointsShift").head(4)
    max_shift = max(team_shift["PointsShift"].abs().max(), 0.1)

    def _block(frame: pd.DataFrame, accent: str, label: str) -> str:
        rows = []
        for _, row in frame.iterrows():
            width = abs(float(row["PointsShift"])) / max_shift * 100
            sign = "+" if row["PointsShift"] >= 0 else ""
            rows.append(
                f"""
                <div class="shift-row" style="--shift-color:{accent};--team-color:{team_color(row['Team'])}">
                  <div class="team-row-head">
                    <div class="team-row-name">
                      <span class="team-swatch" style="background:{team_color(row['Team'])}"></span>
                      <span>{html.escape(row['Team'])}</span>
                    </div>
                    <div class="team-row-value" style="color:{accent}">{sign}{row['PointsShift']:.1f} pts/race</div>
                  </div>
                  <div class="prob-bar slim"><span style="width:{width:.1f}%;background:{accent}"></span></div>
                </div>
                """
            )
        return f"<div class='subsection-label'>{label}</div>{''.join(rows)}"

    return _block(pos, "#59F8B2", "Fastest risers") + _block(neg, "#FF667A", "Biggest fallers")


def render_prediction_cards(sample: pd.DataFrame) -> str:
    cards = []
    for _, row in sample.head(10).iterrows():
        cards.append(
            f"""
            <article class="forecast-card" style="--team-color:{team_color(row['Team'])}">
              <div class="forecast-topline">
                <span class="forecast-rank">P{int(row['PredictedPos'])}</span>
                <span class="forecast-ci">Win CI {fmt_pct(row['WinProb_CI_low'])} to {fmt_pct(row['WinProb_CI_high'])}</span>
              </div>
              <div class="forecast-name">
                <span class="team-swatch" style="background:{team_color(row['Team'])}"></span>
                <div>
                  <div class="driver-name">{html.escape(row['DriverFull'])}</div>
                  <div class="table-meta">{html.escape(row['Team'])}</div>
                </div>
              </div>
              <div class="forecast-metrics">
                <div>
                  <div class="metric-value">{fmt_pct(row['WinProb'])}</div>
                  <div class="metric-label">Win</div>
                </div>
                <div>
                  <div class="metric-value">{fmt_pct(row['PodiumProb'])}</div>
                  <div class="metric-label">Podium</div>
                </div>
                <div>
                  <div class="metric-value">{fmt_pct(row['Top5Prob'])}</div>
                  <div class="metric-label">Top 5</div>
                </div>
              </div>
              <div class="mini-bar-group">
                <div class="mini-bar-row"><span>Top 5</span><div class="prob-bar slim"><span style="width:{row['Top5Prob'] * 100:.1f}%;background:#28C7FA"></span></div></div>
                <div class="mini-bar-row"><span>DNF</span><div class="prob-bar slim"><span style="width:{row['DNFProb'] * 100:.1f}%;background:#FF667A"></span></div></div>
              </div>
            </article>
            """
        )
    return "".join(cards)


def render_feature_rows(features: pd.DataFrame) -> str:
    top = features.head(8).copy()
    top["abs_corr"] = top["corr_with_finish"].abs()
    max_corr = max(float(top["abs_corr"].max()), 0.01)
    rows = []
    accents = ["#FF6B35", "#28C7FA", "#FFD166", "#59F8B2", "#FF8A47", "#66D9FF", "#FFE28A", "#7FF8C1"]
    for idx, (_, row) in enumerate(top.iterrows()):
        width = float(row["abs_corr"]) / max_corr * 100
        rows.append(
            f"""
            <div class="feature-row">
              <div class="feature-head">
                <span>{html.escape(row['feature'])}</span>
                <span>{row['abs_corr']:.3f}</span>
              </div>
              <div class="prob-bar slim"><span style="width:{width:.1f}%;background:{accents[idx % len(accents)]}"></span></div>
            </div>
            """
        )
    return "".join(rows)


def render_validation_cards(metrics: dict[str, dict]) -> str:
    ordered_models = ["ensemble", "quali_baseline", "season_points_baseline"]
    cards = []
    for model_name in ordered_models:
        row = metrics[model_name]
        cards.append(
            f"""
            <article class="model-card {'highlight' if model_name == 'ensemble' else ''}">
              <div class="subsection-label">{MODEL_LABELS[model_name]}</div>
              <div class="model-grid">
                <div>
                  <div class="metric-value">{row['spearman_rho_mean']:.3f}</div>
                  <div class="metric-label">Mean Spearman</div>
                </div>
                <div>
                  <div class="metric-value">{fmt_pct(row['win_accuracy'])}</div>
                  <div class="metric-label">Winner hit rate</div>
                </div>
                <div>
                  <div class="metric-value">{row['mae_positions_mean']:.2f}</div>
                  <div class="metric-label">Mean absolute error</div>
                </div>
              </div>
              <div class="mini-bar-group">
                <div class="mini-bar-row"><span>Rank quality</span><div class="prob-bar slim"><span style="width:{row['spearman_rho_mean'] * 100:.1f}%;background:#59F8B2"></span></div></div>
                <div class="mini-bar-row"><span>Winner calls</span><div class="prob-bar slim"><span style="width:{row['win_accuracy'] * 100:.1f}%;background:#FFD166"></span></div></div>
              </div>
              <div class="range-note">{int(row['Races'])} walk-forward races | podium overlap {row['podium_overlap_mean']:.2f}</div>
            </article>
            """
        )
    return "".join(cards)


def render_watchlist_items() -> str:
    items = []
    for title, body in WATCHLIST_ITEMS:
        items.append(
            f"""
            <div class="watch-item">
              <div class="subsection-label">{html.escape(title)}</div>
              <p>{html.escape(body)}</p>
            </div>
            """
        )
    return "".join(items)


def load_snapshot(source_root: Path) -> dict:
    config = load_config_module(source_root)

    processed_dir = source_root / "data" / "processed"
    results_dir = source_root / "results"
    diagnostics_dir = results_dir / "diagnostics"
    eda_dir = results_dir / "eda"
    ensure_eda_inputs(source_root)

    wdc = pd.read_csv(results_dir / "wdc_forecast_2026.csv")
    wcc = pd.read_csv(results_dir / "wcc_forecast_2026.csv")
    latest_pred_path, latest_meta_path = latest_prediction_paths(source_root)
    latest_meta: dict = {}
    if latest_pred_path.exists():
        sample = pd.read_csv(latest_pred_path)
        if latest_meta_path.exists():
            latest_meta = json.loads(latest_meta_path.read_text(encoding="utf-8"))
    else:
        sample = pd.read_csv(diagnostics_dir / "sample_prediction.csv")
    feature_corr = pd.read_csv(eda_dir / "feature_target_correlation.csv")
    team_shift = pd.read_csv(eda_dir / "live_2026_team_shift.csv")
    diagnostics_summary = json.loads((diagnostics_dir / "summary.json").read_text())

    live_results = pd.read_parquet(processed_dir / "2026_live_results.parquet")
    live_sprint = pd.read_parquet(processed_dir / "2026_live_sprint.parquet")
    constructors = pd.read_parquet(processed_dir / "2026_live_constructor_state.parquet").sort_values("Position")
    training = pd.read_parquet(processed_dir / "master_training_data.parquet")
    feature_matrix = pd.read_parquet(processed_dir / "feature_matrix.parquet")

    driver_lookup = (
        live_results[["Driver", "DriverFull", "Team"]]
        .drop_duplicates("Driver")
        .set_index("Driver")
    )

    race_points = live_results.groupby(["Driver", "Team"], as_index=False)["Points"].sum()
    sprint_points = live_sprint.groupby(["Driver", "Team"], as_index=False)["SprintPoints"].sum()
    driver_standings = race_points.merge(sprint_points, on=["Driver", "Team"], how="left").fillna({"SprintPoints": 0})
    driver_standings["TotalPoints"] = driver_standings["Points"] + driver_standings["SprintPoints"]

    live_stats = live_results.groupby("Driver").agg(
        Wins=("FinishPosition", lambda s: int((s == 1).sum())),
        Podiums=("FinishPosition", lambda s: int((s <= 3).sum())),
        AvgFinish=("FinishPosition", "mean"),
    ).reset_index()

    form = (
        live_results.sort_values(["Driver", "Round"])
        .groupby("Driver")["FinishPosition"]
        .apply(lambda s: " ".join(f"P{int(v)}" for v in s.tail(3)))
        .reset_index(name="Last3")
    )

    driver_standings = (
        driver_standings.merge(live_stats, on="Driver", how="left")
        .merge(form, on="Driver", how="left")
        .merge(driver_lookup[["DriverFull"]], left_on="Driver", right_index=True, how="left")
        .sort_values(["TotalPoints", "Wins", "Podiums", "AvgFinish"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    driver_standings.insert(0, "Position", driver_standings.index + 1)

    wdc["DriverFull"] = wdc["Driver"].map(driver_lookup["DriverFull"]).fillna(wdc["Driver"])
    sample["DriverFull"] = sample["Driver"].map(driver_lookup["DriverFull"]).fillna(sample["Driver"])

    metrics = {
        row["Model"]: row
        for row in diagnostics_summary["baseline_comparison"]
    }

    points_leader = driver_standings.iloc[0]
    title_favorite = wdc.sort_values("WDC_Prob", ascending=False).iloc[0]
    constructor_leader = constructors.iloc[0]

    completed_rounds = int(constructors["CompletedRounds"].max())
    next_round = completed_rounds + 1
    next_race = next(
        (race for race in config.ACTIVE_CIRCUITS_2026 if race["round"] == next_round),
        config.ACTIVE_CIRCUITS_2026[-1],
    )
    next_race_name = gp_name(next_race["name"])
    forecast_race_name = gp_name(latest_meta.get("race_name", next_race["name"]))

    top_prediction = sample.sort_values("PredictedPos").iloc[0]
    wcc_favorite = wcc.sort_values("WCC_Prob", ascending=False).iloc[0]

    summary = {
        "source_root": source_root,
        "generated_at": datetime.now(),
        "training_rows": int(training.shape[0]),
        "feature_columns": int(feature_matrix.shape[1]),
        "train_year_start": int(training["Year"].min()),
        "train_year_end": int(training["Year"].max()),
        "completed_rounds": completed_rounds,
        "next_race_name": next_race_name,
        "forecast_race_name": forecast_race_name,
        "points_leader": points_leader,
        "title_favorite": title_favorite,
        "constructor_leader": constructor_leader,
        "wcc_favorite": wcc_favorite,
        "wcc_gap": float(constructors.iloc[0]["Points"] - constructors.iloc[1]["Points"]),
        "top_prediction": top_prediction,
        "wdc": wdc,
        "wcc": wcc,
        "sample": sample.sort_values("PredictedPos"),
        "feature_corr": feature_corr.sort_values("corr_with_finish", ascending=False),
        "team_shift": team_shift.sort_values("PointsShift", ascending=False),
        "driver_standings": driver_standings,
        "constructors": constructors,
        "metrics": metrics,
        "ensemble_weights": getattr(config, "ENSEMBLE_WEIGHTS", {}),
        "monte_carlo_sims": int(getattr(config, "MONTE_CARLO_N_SIMS", 0)),
        "latest_prediction_meta": latest_meta,
    }
    return summary


def render_html(snapshot: dict) -> str:
    points_leader = snapshot["points_leader"]
    title_favorite = snapshot["title_favorite"]
    constructor_leader = snapshot["constructor_leader"]
    top_prediction = snapshot["top_prediction"]
    metrics = snapshot["metrics"]
    ensemble = metrics["ensemble"]
    quali = metrics["quali_baseline"]
    season = metrics["season_points_baseline"]
    weights = snapshot["ensemble_weights"]
    driver_standings = snapshot["driver_standings"]
    constructors = snapshot["constructors"]
    wdc_sorted = snapshot["wdc"].sort_values("WDC_Prob", ascending=False).reset_index(drop=True)

    standings_runner_up = driver_standings.iloc[1] if len(driver_standings) > 1 else points_leader
    title_chaser = wdc_sorted.iloc[1] if len(wdc_sorted) > 1 else title_favorite
    constructor_runner_up = constructors.iloc[1] if len(constructors) > 1 else constructor_leader
    live_gap = max(0.0, float(points_leader["TotalPoints"] - standings_runner_up["TotalPoints"]))
    title_margin = max(0.0, float(title_favorite["WDC_Prob"] - title_chaser["WDC_Prob"]))

    if points_leader["DriverFull"] != title_favorite["DriverFull"]:
        operating_takeaway = (
            f"{html.escape(points_leader['DriverFull'])} leads the live table by {fmt_num(live_gap)} points over "
            f"{html.escape(standings_runner_up['DriverFull'])}, but the simulator still prefers "
            f"{html.escape(title_favorite['DriverFull'])} over the full season horizon."
        )
    else:
        operating_takeaway = (
            f"{html.escape(points_leader['DriverFull'])} is both the live leader and the probabilistic favourite, "
            "which suggests the current order is already hardening rather than being driven by one noisy weekend."
        )

    constructor_note = (
        f"{html.escape(constructor_leader['Team'])} is {fmt_num(snapshot['wcc_gap'])} points clear of "
        f"{html.escape(constructor_runner_up['Team'])} and already owns "
        f"{fmt_pct(snapshot['wcc_favorite']['WCC_Prob'])} of the WCC book."
    )
    race_note = (
        f"{html.escape(top_prediction['DriverFull'])} is the single most likely winner for "
        f"{html.escape(snapshot['forecast_race_name'])}, but the front of the grid is still compressed with no "
        f"driver above {fmt_pct(top_prediction['WinProb'])} win odds."
    )

    css = """
    :root{
      --bg:#f3eee4;
      --paper:#fbf8f2;
      --paper-2:#efe7d8;
      --ink:#161a21;
      --muted:#6f7480;
      --line:rgba(22,26,33,0.10);
      --line-strong:rgba(22,26,33,0.18);
      --accent:#d72638;
      --accent-2:#1f57d2;
      --amber:#bb7f14;
      --green:#17815f;
      --red:#c14953;
      --coal:#121721;
      --coal-2:#1b2230;
      --display:'Teko',sans-serif;
      --body:'Manrope',sans-serif;
      --mono:'JetBrains Mono',monospace;
      --shadow:0 26px 60px rgba(42,32,20,0.10);
    }
    *{box-sizing:border-box;margin:0;padding:0}
    html{scroll-behavior:smooth}
    body{
      font-family:var(--body);
      color:var(--ink);
      background:
        radial-gradient(circle at 16% 10%, rgba(215,38,56,0.10), transparent 24%),
        radial-gradient(circle at 84% 16%, rgba(31,87,210,0.10), transparent 26%),
        linear-gradient(180deg,#f8f4ec 0%,#f3eee4 54%,#efe7d8 100%);
      line-height:1.5;
    }
    body::before{
      content:"";
      position:fixed;
      inset:0;
      pointer-events:none;
      background:
        repeating-linear-gradient(135deg, rgba(22,26,33,0.015) 0 1px, transparent 1px 14px),
        linear-gradient(180deg, rgba(255,255,255,0.30), rgba(255,255,255,0));
      opacity:.75;
    }
    a{color:inherit}
    .wrap{width:min(1380px,calc(100% - 48px));margin:0 auto}
    .page-shell{position:relative;z-index:1}
    .eyebrow,.slabel,.subsection-label,.metric-label,.table-meta,.range-note,.chip,.footer-note,th,.section-note,.telemetry-label{font-family:var(--mono);letter-spacing:.11em;text-transform:uppercase;font-size:11px}
    .masthead{padding:40px 0 28px}
    .mast-grid{display:grid;grid-template-columns:1.18fr .82fr;gap:22px;align-items:stretch}
    .hero-card,.telemetry-card,.kpi,.board,.callout-panel{border:1px solid var(--line);box-shadow:var(--shadow)}
    .hero-card{
      position:relative;
      background:linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.52));
      border-radius:30px;
      padding:34px 36px 30px;
      overflow:hidden;
    }
    .hero-card::before{
      content:"";
      position:absolute;
      inset:0 auto auto 0;
      width:100%;
      height:10px;
      background:linear-gradient(90deg,var(--accent) 0%, var(--accent) 36%, transparent 36%, transparent 42%, var(--accent-2) 42%, var(--accent-2) 62%, transparent 62%);
    }
    .hero-card::after{
      content:"";
      position:absolute;
      right:-38px;
      bottom:-40px;
      width:220px;
      height:220px;
      border:22px solid rgba(31,87,210,0.08);
      border-radius:50%;
    }
    .eyebrow{color:var(--accent);margin-bottom:16px}
    .hero-kicker{
      display:inline-flex;
      align-items:center;
      gap:10px;
      padding:7px 12px;
      border-radius:999px;
      background:rgba(215,38,56,0.08);
      color:var(--accent);
      font-family:var(--mono);
      letter-spacing:.12em;
      text-transform:uppercase;
      font-size:11px;
      margin-bottom:16px;
    }
    .hero-title{
      font-family:var(--display);
      font-size:clamp(82px,10.5vw,148px);
      line-height:.78;
      letter-spacing:.03em;
      margin-bottom:18px;
    }
    .hero-copy{
      max-width:760px;
      color:#454c58;
      font-size:15px;
      line-height:1.9;
      margin-bottom:24px;
    }
    .chip-row{display:flex;flex-wrap:wrap;gap:10px}
    .chip{
      border:1px solid var(--line);
      border-radius:999px;
      padding:9px 12px;
      color:#5d6370;
      background:rgba(255,255,255,0.54);
    }
    .chip.hot{color:var(--accent);border-color:rgba(215,38,56,0.22);background:rgba(215,38,56,0.08)}
    .chip.cool{color:var(--accent-2);border-color:rgba(31,87,210,0.18);background:rgba(31,87,210,0.08)}
    .chip.good{color:var(--green);border-color:rgba(23,129,95,0.18);background:rgba(23,129,95,0.08)}
    .telemetry-card{
      background:linear-gradient(180deg,var(--coal) 0%, var(--coal-2) 100%);
      color:#f2f5f8;
      border-radius:30px;
      padding:28px;
      position:relative;
      overflow:hidden;
    }
    .telemetry-card::before{
      content:"";
      position:absolute;
      inset:0 0 auto 0;
      height:8px;
      background:linear-gradient(90deg,var(--accent),var(--accent-2));
    }
    .telemetry-label{color:#8ea4ff;margin-bottom:14px}
    .telemetry-card h2{
      font-family:var(--display);
      font-size:54px;
      line-height:.82;
      letter-spacing:.04em;
      margin-bottom:18px;
    }
    .telemetry-note{font-size:13px;line-height:1.85;color:#bcc6d7;margin-top:18px}
    .telemetry-stat{padding:14px 0;border-top:1px solid rgba(255,255,255,0.10)}
    .telemetry-stat strong{display:block;font-size:24px;line-height:1.3;margin-top:4px}
    .metric-band{padding:8px 0 0}
    .kpi-rack{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
    .kpi{
      background:rgba(255,255,255,0.62);
      backdrop-filter:blur(8px);
      border-radius:24px;
      padding:22px 22px 20px;
      position:relative;
      overflow:hidden;
    }
    .kpi::before{
      content:"";
      position:absolute;
      inset:0 auto auto 0;
      width:100%;
      height:5px;
      background:linear-gradient(90deg,var(--accent), transparent 75%);
    }
    .kpi:nth-child(2)::before{background:linear-gradient(90deg,var(--accent-2), transparent 75%)}
    .kpi:nth-child(3)::before{background:linear-gradient(90deg,var(--green), transparent 75%)}
    .kpi:nth-child(4)::before{background:linear-gradient(90deg,var(--amber), transparent 75%)}
    .kpi-value{font-family:var(--display);font-size:56px;line-height:.84;letter-spacing:.03em;margin-bottom:4px}
    .kpi-copy{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.7}
    .callout-wrap{padding:18px 0 6px}
    .callout-panel{
      background:linear-gradient(90deg, rgba(22,26,33,0.98) 0%, rgba(35,43,58,0.96) 100%);
      color:#f4f7fb;
      border-radius:24px;
      padding:18px 22px;
      display:grid;
      grid-template-columns:.32fr 1fr;
      gap:18px;
      align-items:center;
    }
    .callout-label{
      font-family:var(--mono);
      letter-spacing:.12em;
      text-transform:uppercase;
      font-size:11px;
      color:#8ea4ff;
    }
    .callout-copy{font-size:15px;line-height:1.8;color:#d6dce7}
    .section{padding:28px 0}
    .main-grid,.split-grid,.analysis-grid,.story-grid{display:grid;gap:18px}
    .main-grid{grid-template-columns:1.08fr .92fr}
    .split-grid{grid-template-columns:.95fr 1.05fr}
    .analysis-grid{grid-template-columns:1fr 1fr 1fr}
    .board{
      position:relative;
      background:rgba(255,255,255,0.62);
      backdrop-filter:blur(8px);
      border-radius:28px;
      padding:28px;
      overflow:hidden;
    }
    .board::before{
      content:"";
      position:absolute;
      inset:0 auto auto 0;
      width:100%;
      height:6px;
      background:linear-gradient(90deg,var(--accent), transparent 72%);
    }
    .board-table::before{background:linear-gradient(90deg,var(--accent-2), transparent 72%)}
    .board-secondary::before{background:linear-gradient(90deg,var(--green), transparent 72%)}
    .board-alert::before{background:linear-gradient(90deg,var(--amber), transparent 72%)}
    .board-dark{
      background:linear-gradient(180deg,#151b26 0%, #1d2430 100%);
      color:#eef3f8;
      border-color:rgba(0,0,0,0);
    }
    .board-dark::before{background:linear-gradient(90deg,var(--accent),var(--accent-2))}
    .board-dark .copy,.board-dark .range-note,.board-dark .metric-label,.board-dark .subsection-label{color:#bcc6d7}
    .board-dark .slabel{color:#f0f4fa}
    .slabel{display:flex;align-items:center;gap:10px;color:var(--accent);margin-bottom:14px}
    .slabel::before{content:"";width:10px;height:10px;border-radius:50%;background:currentColor}
    .ctitle{font-family:var(--display);font-size:50px;line-height:.82;letter-spacing:.04em;margin-bottom:10px}
    .copy{font-size:13px;color:var(--muted);line-height:1.85}
    .divider{height:1px;background:var(--line);margin:20px 0}
    .board-dark .divider{background:rgba(255,255,255,0.10)}
    .prob-row,.feature-row,.watch-item,.shift-row,.team-row{margin-bottom:16px}
    .prob-row:last-child,.feature-row:last-child,.watch-item:last-child,.shift-row:last-child,.team-row:last-child{margin-bottom:0}
    .prob-row{
      position:relative;
      padding:14px 16px 14px 18px;
      border:1px solid var(--line);
      border-radius:18px;
      background:rgba(255,255,255,0.52);
    }
    .prob-row::before{
      content:"";
      position:absolute;
      inset:14px auto 14px 0;
      width:4px;
      border-radius:0 999px 999px 0;
      background:var(--team-color, var(--accent));
    }
    .prob-head,.feature-head,.team-row-head,.forecast-topline,.mini-bar-row{display:flex;justify-content:space-between;gap:14px;align-items:center}
    .driver-line,.team-row-name,.forecast-name,.driver-cell{display:flex;align-items:center;gap:10px}
    .driver-cell{gap:12px}
    .driver-name{font-size:15px;font-weight:800}
    .prob-meta{font-size:12px;color:var(--muted);margin-top:4px}
    .prob-value{font-family:var(--display);font-size:40px;color:var(--team-color, var(--accent));white-space:nowrap;line-height:.85}
    .prob-bar{height:8px;border-radius:999px;background:rgba(22,26,33,0.07);overflow:hidden;margin:10px 0 6px}
    .prob-bar.slim{height:6px;margin:8px 0 0}
    .prob-bar span{display:block;height:100%;border-radius:999px}
    .team-swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0;box-shadow:0 0 0 4px rgba(22,26,33,0.04)}
    .rank-chip{width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(22,26,33,0.06);color:var(--muted);font-family:var(--mono);font-size:10px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:12px 10px;border-bottom:1px solid var(--line);text-align:left}
    th{color:var(--muted)}
    td{font-size:13px;vertical-align:middle}
    td:nth-child(1),td:nth-child(3),td:nth-child(4),td:nth-child(5),td:nth-child(6){text-align:right}
    td:nth-child(7){min-width:118px}
    tr:hover td{background:rgba(255,255,255,0.52)}
    .table-wrap{overflow:auto}
    .form-pill{display:inline-flex;align-items:center;justify-content:center;width:30px;height:24px;border-radius:8px;font-family:var(--mono);font-size:10px;margin-right:6px}
    .form-pill.good{background:rgba(23,129,95,0.12);color:var(--green)}
    .form-pill.mid{background:rgba(187,127,20,0.12);color:var(--amber)}
    .form-pill.bad{background:rgba(193,73,83,0.12);color:var(--red)}
    .section-head{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:18px}
    .section-note{color:var(--muted);max-width:420px;line-height:1.8}
    .forecast-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}
    .forecast-card{
      position:relative;
      background:rgba(255,255,255,0.66);
      border:1px solid var(--line);
      border-radius:20px;
      padding:18px;
      overflow:hidden;
    }
    .forecast-card::before{
      content:"";
      position:absolute;
      inset:0 0 auto 0;
      height:4px;
      background:var(--team-color, var(--accent));
    }
    .forecast-rank{font-family:var(--display);font-size:34px;letter-spacing:.04em;color:var(--team-color, var(--accent));line-height:.88}
    .forecast-ci{font-size:10px;color:var(--muted);font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em;text-align:right}
    .forecast-metrics,.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0 16px}
    .metric-value{font-family:var(--display);font-size:34px;line-height:.86;letter-spacing:.03em}
    .mini-bar-group{display:grid;gap:10px}
    .mini-bar-row span:first-child{font-size:11px;color:var(--muted);font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em}
    .mini-bar-row .prob-bar{flex:1;margin:0 0 0 12px}
    .subsection-label{color:var(--accent-2);margin-bottom:10px}
    .team-row,.shift-row{
      padding:12px 14px;
      border:1px solid var(--line);
      border-radius:16px;
      background:rgba(255,255,255,0.48);
    }
    .team-row-value{font-weight:700}
    .shift-row .team-row-value{color:var(--shift-color, var(--accent))}
    .feature-head span:first-child{font-weight:700}
    .model-card{
      background:rgba(255,255,255,0.48);
      border:1px solid var(--line);
      border-radius:18px;
      padding:18px;
      margin-bottom:14px;
    }
    .model-card:last-child{margin-bottom:0}
    .model-card.highlight{
      border-color:rgba(23,129,95,0.22);
      background:linear-gradient(180deg, rgba(23,129,95,0.10), rgba(255,255,255,0.48));
    }
    .watch-item p{font-size:13px;color:var(--muted);line-height:1.8}
    .memo-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:20px}
    .memo-tile{
      background:rgba(255,255,255,0.06);
      border:1px solid rgba(255,255,255,0.10);
      border-radius:18px;
      padding:16px;
    }
    .memo-value{font-family:var(--display);font-size:42px;line-height:.86;letter-spacing:.03em;margin:8px 0 6px}
    .memo-tile p{font-size:12px;color:#c0cada;line-height:1.8}
    .story-band{
      background:linear-gradient(135deg,#151b26 0%, #1c2431 58%, #10151e 100%);
      border-top:1px solid rgba(0,0,0,0.05);
      border-bottom:1px solid rgba(0,0,0,0.05);
      padding:46px 0;
      margin-top:8px;
    }
    .story-grid{grid-template-columns:1.12fr .88fr;align-items:start}
    .story-copy{padding-right:8px}
    .story-copy p{color:#c0cada;font-size:14px;line-height:1.9;margin-bottom:16px}
    .story-copy strong{color:#f2f5f8}
    .slabel-invert{color:#8ea4ff}
    .story-title{font-family:var(--display);font-size:58px;line-height:.82;letter-spacing:.04em;color:#f4f7fb;margin-bottom:16px}
    .method-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .method-card{
      background:rgba(255,255,255,0.05);
      border:1px solid rgba(255,255,255,0.10);
      border-radius:18px;
      padding:18px;
    }
    .method-card p{font-size:13px;color:#c0cada;line-height:1.8}
    .footer{padding:28px 0 42px}
    .footer-grid{display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
    .footer-grid strong{font-family:var(--display);font-size:28px;color:var(--ink);letter-spacing:.03em}
    .footer-right{text-align:right}
    @media (max-width:1240px){
      .mast-grid,.main-grid,.split-grid,.analysis-grid,.story-grid{grid-template-columns:1fr}
      .forecast-grid{grid-template-columns:repeat(3,1fr)}
      .callout-panel{grid-template-columns:1fr}
      .memo-grid{grid-template-columns:1fr 1fr}
    }
    @media (max-width:900px){
      .wrap{width:min(100% - 28px,1380px)}
      .kpi-rack,.forecast-grid,.method-grid,.memo-grid{grid-template-columns:1fr}
      .section-head{flex-direction:column;align-items:flex-start}
      .hero-title{font-size:72px}
      .telemetry-card h2,.ctitle,.story-title{font-size:48px}
      table{display:block;overflow:auto}
      .footer-right{text-align:left}
    }
    """

    methodology_cards = [
        (
            "Grouped race ranker",
            f"XGBoost learns the order inside each grand prix rather than just a binary win signal, using {snapshot['training_rows']:,} driver-race rows from {snapshot['train_year_start']} to {snapshot['train_year_end']}.",
        ),
        (
            "Pairwise driver strength",
            "Bradley-Terry estimates head-to-head driver advantage, giving the stack a driver-vs-driver layer that pure table features usually miss.",
        ),
        (
            "Monte Carlo season engine",
            f"{snapshot['monte_carlo_sims']:,} simulations translate point estimates into title odds, podium ranges, and season-end distributions rather than false certainty.",
        ),
        (
            "Weighted ensemble",
            f"Final forecast blend is XGBoost {weights.get('xgboost', 0):.0%}, Bradley-Terry {weights.get('bradley_terry', 0):.0%}, Monte Carlo {weights.get('monte_carlo', 0):.0%}.",
        ),
    ]
    methodology_html = "".join(
        f"""
        <div class="method-card">
          <div class="subsection-label">{html.escape(title)}</div>
          <p>{html.escape(body)}</p>
        </div>
        """
        for title, body in methodology_cards
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>F1 Predictor | 2026 Portfolio Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Manrope:wght@400;500;700;800&family=Teko:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <style>{css}</style>
</head>
<body>
  <div class="page-shell">
    <header class="masthead">
      <div class="wrap">
        <div class="mast-grid">
          <section class="hero-card">
            <div class="eyebrow">Executive race briefing | Formula 1 2026 | Snapshot {fmt_date(snapshot['generated_at'])}</div>
            <div class="hero-kicker">Pit Wall Dossier</div>
            <h1 class="hero-title">CHAMPIONSHIP<br/>CONTROL ROOM</h1>
            <p class="hero-copy">A consulting-grade motorsport briefing designed for portfolio presentation: live championship table, probabilistic title map, and race-week scenario board in one sheet. The forecasting stack blends grouped XGBoost ranking, Bradley-Terry driver strength, constructor state, and Monte Carlo simulation so the page reads like an operating review rather than a fan-only scoreboard.</p>
            <div class="chip-row">
              <span class="chip hot">{snapshot['completed_rounds']} rounds complete | next race {html.escape(snapshot['next_race_name'])}</span>
              <span class="chip cool">{snapshot['training_rows']:,} driver-race rows | {snapshot['feature_columns']} model columns</span>
              <span class="chip good">{snapshot['monte_carlo_sims']:,} Monte Carlo season sims</span>
              <span class="chip">{MODEL_LABELS['ensemble']} leads on race-order quality</span>
              <span class="chip">Live leader {html.escape(points_leader['DriverFull'])} | forecast leader {html.escape(title_favorite['DriverFull'])}</span>
            </div>
          </section>
          <aside class="telemetry-card">
            <div class="telemetry-label">Trackside note</div>
            <h2>THE TABLE IS A LAGGING INDICATOR</h2>
            <div class="telemetry-stat">
              <div class="metric-label">Current points leader</div>
              <strong>{html.escape(points_leader['DriverFull'])} | {fmt_num(points_leader['TotalPoints'])} pts</strong>
            </div>
            <div class="telemetry-stat">
              <div class="metric-label">Forecast title favourite</div>
              <strong>{html.escape(title_favorite['DriverFull'])} | {fmt_pct(title_favorite['WDC_Prob'])} WDC probability</strong>
            </div>
            <div class="telemetry-stat">
              <div class="metric-label">Constructor control</div>
              <strong>{html.escape(constructor_leader['Team'])} | {fmt_pct(snapshot['wcc_favorite']['WCC_Prob'])} WCC probability</strong>
            </div>
            <p class="telemetry-note">{operating_takeaway}</p>
          </aside>
        </div>
      </div>
    </header>

    <section class="metric-band">
      <div class="wrap kpi-rack">
        <article class="kpi">
          <div class="kpi-value" style="color:var(--accent)">{fmt_pct(title_favorite['WDC_Prob'])}</div>
          <div class="metric-label">{html.escape(title_favorite['DriverFull'])} title probability</div>
          <div class="kpi-copy">{fmt_pct(title_margin)} probability edge over {html.escape(title_chaser['DriverFull'])}</div>
        </article>
        <article class="kpi">
          <div class="kpi-value" style="color:var(--accent-2)">{ensemble['spearman_rho_mean']:.3f}</div>
          <div class="metric-label">Walk-forward rank quality</div>
          <div class="kpi-copy">{int(ensemble['Races'])} historical races scored with no future leakage</div>
        </article>
        <article class="kpi">
          <div class="kpi-value" style="color:var(--green)">{fmt_pct(snapshot['wcc_favorite']['WCC_Prob'])}</div>
          <div class="metric-label">{html.escape(snapshot['wcc_favorite']['Team'])} WCC probability</div>
          <div class="kpi-copy">{fmt_num(snapshot['wcc_gap'])} point margin over {html.escape(constructor_runner_up['Team'])}</div>
        </article>
        <article class="kpi">
          <div class="kpi-value" style="color:var(--amber)">{fmt_pct(top_prediction['WinProb'])}</div>
          <div class="metric-label">{html.escape(top_prediction['DriverFull'])} race-win chance</div>
          <div class="kpi-copy">{html.escape(snapshot['forecast_race_name'])} forecast remains wide open at the front</div>
        </article>
      </div>
    </section>

    <section class="callout-wrap">
      <div class="wrap">
        <div class="callout-panel">
          <div class="callout-label">Management summary</div>
          <div class="callout-copy">{operating_takeaway} {constructor_note}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="wrap main-grid">
        <article class="board">
          <div class="slabel">Title map</div>
          <div class="ctitle">WDC PROBABILITY LADDER</div>
          <p class="copy">Expected final points determine bar length; title probability sits on the right. That keeps the visual useful for leadership audiences even when the championship compresses into a small number of viable contenders.</p>
          <div class="divider"></div>
          {render_wdc_rows(snapshot['wdc'])}
        </article>

        <article class="board board-table">
          <div class="slabel">Live order</div>
          <div class="ctitle">DRIVER STANDINGS</div>
          <p class="copy">Current 2026 points through round {snapshot['completed_rounds']}, including sprint scoring. Last-three finish tokens keep momentum visible without drowning the page in lap-level detail.</p>
          <div class="divider"></div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Driver</th>
                  <th>Pts</th>
                  <th>Wins</th>
                  <th>Podiums</th>
                  <th>Avg</th>
                  <th>Last 3</th>
                </tr>
              </thead>
              <tbody>{render_driver_table(driver_standings)}</tbody>
            </table>
          </div>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="wrap split-grid">
        <article class="board board-secondary">
          <div class="slabel">Team control</div>
          <div class="ctitle">CONSTRUCTOR ORDER</div>
          <p class="copy">Current constructor points through round {snapshot['completed_rounds']}. The table is shown alongside pace-reset movement so you can separate present scoring from adaptation speed under the new regulations.</p>
          <div class="divider"></div>
          {render_constructor_rows(constructors)}
          <div class="divider"></div>
          <div class="subsection-label">Reset winners and losers</div>
          <div class="range-note">Points delta per race versus the 2025 baseline</div>
          <div style="margin-top:14px">{render_shift_rows(snapshot['team_shift'])}</div>
        </article>

        <article class="board board-dark">
          <div class="slabel">Decision memo</div>
          <div class="ctitle">WHERE THE MODEL EARNS TRUST</div>
          <p class="copy">This page is built to be useful in a consulting conversation: not just who is ahead, but what is stable, what is volatile, and where the data story is intentionally separated from the current headline standings.</p>
          <div class="memo-grid">
            <div class="memo-tile">
              <div class="subsection-label">Standings tension</div>
              <div class="memo-value">{fmt_num(live_gap)}</div>
              <p>Points between {html.escape(points_leader['DriverFull'])} and {html.escape(standings_runner_up['DriverFull'])}. The live table is close enough that a few weekends can still reshape the conversation.</p>
            </div>
            <div class="memo-tile">
              <div class="subsection-label">Constructor margin</div>
              <div class="memo-value">{fmt_pct(snapshot['wcc_favorite']['WCC_Prob'])}</div>
              <p>{constructor_note}</p>
            </div>
            <div class="memo-tile">
              <div class="subsection-label">Race-week volatility</div>
              <div class="memo-value">{fmt_pct(top_prediction['WinProb'])}</div>
              <p>{race_note}</p>
            </div>
          </div>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="wrap">
        <div class="section-head">
          <div>
            <div class="slabel">Race week board</div>
            <div class="ctitle">{html.escape(snapshot['forecast_race_name'].upper())} SCENARIO BOARD</div>
          </div>
          <div class="section-note">Top projected finishers from the latest archived prediction. Each tile combines predicted finishing slot with win, podium, top-five and DNF probabilities, so pace and reliability are visible together.</div>
        </div>
        <div class="forecast-grid">{render_prediction_cards(snapshot['sample'])}</div>
      </div>
    </section>

    <section class="section">
      <div class="wrap analysis-grid">
        <article class="board">
          <div class="slabel">Signal audit</div>
          <div class="ctitle">WHAT MOVES THE ORDER</div>
          <p class="copy">Absolute correlation with finish position from the saved feature matrix. These are structural signals visible in the current artifacts, not claims of direct causality.</p>
          <div class="divider"></div>
          {render_feature_rows(snapshot['feature_corr'])}
        </article>

        <article class="board">
          <div class="slabel">Validation</div>
          <div class="ctitle">HONEST BACKTEST</div>
          <p class="copy">Walk-forward only: train on the past and score the future. The ensemble leads on ranking quality, while the qualifying baseline still steals some cleaner single-winner calls.</p>
          <div class="divider"></div>
          {render_validation_cards(snapshot['metrics'])}
        </article>

        <article class="board board-alert">
          <div class="slabel">Known gaps</div>
          <div class="ctitle">MODEL WATCHLIST</div>
          <p class="copy">The page is intentionally transparent about what still needs fixing. That transparency matters when this is shown to operators, recruiters, or leadership teams instead of staying in a notebook.</p>
          <div class="divider"></div>
          {render_watchlist_items()}
        </article>
      </div>
    </section>

    <section class="story-band">
      <div class="wrap story-grid">
        <div class="story-copy">
          <div class="slabel slabel-invert">Method note</div>
          <div class="story-title">WHY THIS LOOKS LIKE A RACE BRIEFING, NOT A SCOREBOARD</div>
          <p><strong>Standings are descriptive.</strong> Forecasts are decision tools. That is why this page deliberately puts the live table next to the WDC probability ladder instead of pretending current points are the same thing as future control.</p>
          <p><strong>The ranking model is the anchor.</strong> Grouped XGBoost learns the relative order inside each grand prix, Bradley-Terry adds head-to-head driver signal, and Monte Carlo converts point estimates into title odds, confidence bands, and scenario ranges.</p>
          <p><strong>Validation stays honest.</strong> Walk-forward evaluation prevents future leakage, and the watchlist remains visible so the dashboard feels executive-grade and technically self-aware at the same time.</p>
        </div>
        <div class="method-grid">{methodology_html}</div>
      </div>
    </section>

    <footer class="footer">
      <div class="wrap footer-grid">
        <div>
          <strong>PIT WALL DOSSIER</strong><br/>
          Data stack: Jolpica + FastF1 + live 2026 parquet layer<br/>
          Forecast stack: XGBoost ranking + Bradley-Terry + Monte Carlo blend<br/>
          Snapshot generated {fmt_date(snapshot['generated_at'])}
        </div>
        <div class="footer-right">
          <div class="footer-note">Portfolio design choices</div>
          Differentiate live standings from forward probabilities<br/>
          Show executive narrative and technical rigor on one page<br/>
          Keep validation leakage-free and visually legible<br/>
          Surface model gaps instead of hiding them
        </div>
      </div>
    </footer>
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an F1 2026 portfolio dashboard HTML file.")
    parser.add_argument("--source-root", help="Path to the F1 predictor repo containing results/ and data/processed/")
    parser.add_argument("--output", help="HTML output path")
    return parser.parse_args()


def generate_dashboard(
    source_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    resolved_root = detect_source_root(str(source_root)) if source_root else detect_source_root(None)
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else latest_dashboard_path(resolved_root)
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(resolved_root)
    html_text = render_html(snapshot)
    resolved_output.write_text(html_text, encoding="utf-8")
    return resolved_output


def main() -> None:
    args = parse_args()
    source_root = detect_source_root(args.source_root) if args.source_root else detect_source_root(None)
    output_path = generate_dashboard(source_root=source_root, output_path=args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dashboard written to {output_path}")
    print(f"Source root: {source_root}")


if __name__ == "__main__":
    main()
