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
        "latest_csv": str(latest_csv),
        "archive_csv": str(archive_csv),
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
            <div class="prob-row">
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
            <div class="team-row">
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
                <div class="shift-row">
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
            <article class="forecast-card">
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

    css = """
    :root{
      --bg:#050915;
      --panel:#0d1525;
      --panel2:#131d33;
      --panel3:#17243f;
      --text:#eef3f8;
      --muted:#8f9db0;
      --line:rgba(255,255,255,0.08);
      --line2:rgba(255,255,255,0.16);
      --accent:#ff6b35;
      --cyan:#28c7fa;
      --amber:#ffd166;
      --green:#59f8b2;
      --red:#ff667a;
      --display:'Bebas Neue',sans-serif;
      --body:'Space Grotesk',sans-serif;
      --mono:'IBM Plex Mono',monospace;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    html{scroll-behavior:smooth}
    body{
      font-family:var(--body);
      color:var(--text);
      background:
        radial-gradient(circle at 15% 15%, rgba(40,199,250,0.10), transparent 28%),
        radial-gradient(circle at 82% 12%, rgba(255,107,53,0.12), transparent 30%),
        radial-gradient(circle at 60% 100%, rgba(89,248,178,0.08), transparent 35%),
        linear-gradient(180deg,#050915 0%,#091121 50%,#050915 100%);
      line-height:1.5;
    }
    body::before{
      content:"";
      position:fixed;
      inset:0;
      pointer-events:none;
      background-image:
        linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
      background-size:36px 36px;
      mask-image:linear-gradient(180deg, rgba(0,0,0,0.5), transparent 90%);
    }
    a{color:inherit}
    .wrap{width:min(1360px,calc(100% - 48px));margin:0 auto}
    .hero{padding:64px 0 44px;border-bottom:1px solid var(--line)}
    .hero-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:24px;align-items:end}
    .eyebrow,.slabel,.subsection-label,.metric-label,.table-meta,.range-note,.chip,.footer-note,th{font-family:var(--mono);letter-spacing:.12em;text-transform:uppercase;font-size:11px}
    .eyebrow{color:var(--cyan);margin-bottom:18px}
    .hero-title{font-family:var(--display);font-size:clamp(66px,10vw,118px);line-height:.88;letter-spacing:.03em;margin-bottom:22px}
    .hero-title span{color:var(--accent)}
    .hero-copy{max-width:760px;color:var(--muted);font-size:15px;line-height:1.9;margin-bottom:26px}
    .chip-row{display:flex;flex-wrap:wrap;gap:10px}
    .chip{border:1px solid var(--line2);border-radius:999px;padding:8px 12px;color:var(--muted);background:rgba(255,255,255,0.03)}
    .chip.hot{color:var(--accent);border-color:rgba(255,107,53,0.35);background:rgba(255,107,53,0.08)}
    .chip.cool{color:var(--cyan);border-color:rgba(40,199,250,0.35);background:rgba(40,199,250,0.08)}
    .chip.good{color:var(--green);border-color:rgba(89,248,178,0.35);background:rgba(89,248,178,0.08)}
    .hero-panel{background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));border:1px solid var(--line);border-radius:24px;padding:24px;backdrop-filter:blur(12px);box-shadow:0 20px 50px rgba(0,0,0,0.22)}
    .hero-panel h3{font-family:var(--display);font-size:40px;letter-spacing:.04em;line-height:.95;margin:10px 0 18px}
    .hero-stat{padding:14px 0;border-top:1px solid var(--line)}
    .hero-stat strong{display:block;font-size:22px;font-weight:700}
    .hero-panel p{font-size:13px;color:var(--muted);line-height:1.8;margin-top:16px}
    .kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);margin:0 auto}
    .kpi{background:rgba(7,12,24,0.82);padding:28px 24px}
    .kpi-value{font-family:var(--display);font-size:52px;line-height:.95;letter-spacing:.03em;margin-bottom:6px}
    .kpi-copy{font-size:12px;color:var(--muted);margin-top:4px}
    .section{padding:34px 0;border-top:1px solid var(--line)}
    .grid-2,.grid-3{display:grid;gap:18px}
    .grid-2{grid-template-columns:1.05fr .95fr}
    .grid-3{grid-template-columns:1fr 1fr 1fr}
    .card{background:linear-gradient(180deg, rgba(19,29,51,0.72), rgba(11,18,32,0.76));border:1px solid var(--line);border-radius:24px;padding:28px;box-shadow:0 22px 50px rgba(0,0,0,0.18);overflow:hidden}
    .slabel{color:var(--muted);display:flex;gap:12px;align-items:center;margin-bottom:16px}
    .slabel::after{content:"";height:1px;flex:1;background:var(--line)}
    .ctitle{font-family:var(--display);font-size:38px;line-height:.92;letter-spacing:.04em;margin-bottom:12px}
    .copy{font-size:13px;color:var(--muted);line-height:1.85}
    .divider{height:1px;background:var(--line);margin:22px 0}
    .prob-row,.feature-row,.watch-item,.shift-row{margin-bottom:16px}
    .prob-head,.feature-head,.team-row-head,.forecast-topline,.mini-bar-row{display:flex;justify-content:space-between;gap:14px;align-items:center}
    .driver-line,.team-row-name,.forecast-name,.driver-cell{display:flex;align-items:center;gap:10px}
    .driver-cell{gap:12px}
    .driver-name{font-size:15px;font-weight:700}
    .prob-meta{font-size:12px;color:var(--muted);margin-top:4px}
    .prob-value{font-family:var(--display);font-size:34px;color:var(--accent);white-space:nowrap}
    .prob-bar{height:8px;border-radius:999px;background:rgba(255,255,255,0.06);overflow:hidden;margin:10px 0 6px}
    .prob-bar.slim{height:6px;margin:8px 0 0}
    .prob-bar span{display:block;height:100%;border-radius:999px}
    .team-swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0;box-shadow:0 0 0 4px rgba(255,255,255,0.03)}
    .rank-chip{width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(255,255,255,0.06);color:var(--muted);font-family:var(--mono);font-size:10px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:12px 10px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:left}
    th{color:var(--muted)}
    td{font-size:13px;vertical-align:middle}
    td:nth-child(1),td:nth-child(3),td:nth-child(4),td:nth-child(5),td:nth-child(6){text-align:right}
    td:nth-child(7){min-width:118px}
    .form-pill{display:inline-flex;align-items:center;justify-content:center;width:30px;height:24px;border-radius:8px;font-family:var(--mono);font-size:10px;margin-right:6px}
    .form-pill.good{background:rgba(89,248,178,0.14);color:var(--green)}
    .form-pill.mid{background:rgba(255,209,102,0.14);color:var(--amber)}
    .form-pill.bad{background:rgba(255,102,122,0.14);color:var(--red)}
    .callout{background:linear-gradient(180deg, rgba(255,107,53,0.10), rgba(255,107,53,0.04));border-color:rgba(255,107,53,0.18)}
    .forecast-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-top:22px}
    .forecast-card{background:rgba(255,255,255,0.03);border:1px solid var(--line);border-radius:20px;padding:18px}
    .forecast-rank{font-family:var(--display);font-size:26px;letter-spacing:.04em;color:var(--cyan)}
    .forecast-ci{font-size:10px;color:var(--muted);font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em}
    .forecast-metrics,.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0 16px}
    .metric-value{font-family:var(--display);font-size:28px;line-height:.95;letter-spacing:.03em}
    .mini-bar-group{display:grid;gap:10px}
    .mini-bar-row span:first-child{font-size:11px;color:var(--muted);font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em}
    .mini-bar-row .prob-bar{flex:1;margin:0 0 0 12px}
    .subsection-label{color:var(--cyan);margin-bottom:10px}
    .model-card{background:rgba(255,255,255,0.03);border:1px solid var(--line);border-radius:18px;padding:18px}
    .model-card.highlight{border-color:rgba(89,248,178,0.25);background:linear-gradient(180deg, rgba(89,248,178,0.08), rgba(255,255,255,0.03))}
    .watch-item p{font-size:13px;color:var(--muted);line-height:1.8}
    .story{background:linear-gradient(180deg, rgba(11,18,32,0.96), rgba(7,12,24,0.96));border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:52px 0;margin-top:8px}
    .story-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:24px;align-items:start}
    .story-copy p{color:var(--muted);font-size:14px;line-height:1.9;margin-bottom:16px}
    .story-copy strong{color:var(--text)}
    .method-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:22px}
    .method-card{background:rgba(255,255,255,0.03);border:1px solid var(--line);border-radius:18px;padding:18px}
    .method-card p{font-size:13px;color:var(--muted);line-height:1.8}
    .footer{padding:30px 0 44px}
    .footer-grid{display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
    .footer-grid strong{font-family:var(--display);font-size:22px;color:var(--text);letter-spacing:.03em}
    @media (max-width:1200px){.hero-grid,.grid-2,.grid-3,.story-grid,.forecast-grid{grid-template-columns:1fr 1fr}.forecast-grid{grid-template-columns:repeat(3,1fr)}}
    @media (max-width:860px){.wrap{width:min(100% - 28px,1360px)}.hero-grid,.grid-2,.grid-3,.story-grid,.forecast-grid,.kpi-strip,.method-grid{grid-template-columns:1fr}.hero-title{font-size:58px}.forecast-grid{grid-template-columns:1fr}table{display:block;overflow:auto}}
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
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet"/>
  <style>{css}</style>
</head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Portfolio ML System | Formula 1 2026 | Snapshot {fmt_date(snapshot['generated_at'])}</div>
          <h1 class="hero-title">FORMULA 1<br/><span>FORECAST ENGINE</span></h1>
          <p class="hero-copy">A consulting-grade racing intelligence dashboard built on top of an ensemble predictor: grouped XGBoost ranking, Bradley-Terry driver strength, championship state, and Monte Carlo uncertainty layered into one executive-facing view. The objective is not just to call winners, but to explain how car order, driver form, and season volatility are shifting under the 2026 reset.</p>
          <div class="chip-row">
            <span class="chip hot">{snapshot['completed_rounds']} rounds complete | next race {html.escape(snapshot['next_race_name'])}</span>
            <span class="chip cool">{snapshot['training_rows']:,} driver-race rows | {snapshot['feature_columns']} model columns</span>
            <span class="chip good">{snapshot['monte_carlo_sims']:,} Monte Carlo season sims</span>
            <span class="chip">{MODEL_LABELS['ensemble']} beats baselines on rank quality</span>
            <span class="chip">Current leader {html.escape(points_leader['DriverFull'])} | model favourite {html.escape(title_favorite['DriverFull'])}</span>
          </div>
        </div>
        <aside class="hero-panel">
          <div class="subsection-label">Why this matters</div>
          <h3>STANDINGS AND FORECASTS ARE NOT THE SAME THING</h3>
          <div class="hero-stat">
            <div class="metric-label">Points leader after round {snapshot['completed_rounds']}</div>
            <strong>{html.escape(points_leader['DriverFull'])} | {fmt_num(points_leader['TotalPoints'])} pts</strong>
          </div>
          <div class="hero-stat">
            <div class="metric-label">Model favourite for the title</div>
            <strong>{html.escape(title_favorite['DriverFull'])} | {fmt_pct(title_favorite['WDC_Prob'])} WDC probability</strong>
          </div>
          <div class="hero-stat">
            <div class="metric-label">Constructor control</div>
            <strong>{html.escape(constructor_leader['Team'])} | {fmt_pct(snapshot['wcc_favorite']['WCC_Prob'])} WCC probability</strong>
          </div>
          <p>The dashboard is designed for upper-management audiences: not just who is ahead, but where the operating picture is changing, where the model is trustworthy, and where the current system still needs improvement.</p>
        </aside>
      </div>
    </div>
  </header>

  <section class="kpi-strip wrap">
    <div class="kpi">
      <div class="kpi-value" style="color:var(--accent)">{fmt_pct(title_favorite['WDC_Prob'])}</div>
      <div class="metric-label">George Russell title probability</div>
      <div class="kpi-copy">Forecast favourite even though Antonelli leads the live table</div>
    </div>
    <div class="kpi">
      <div class="kpi-value" style="color:var(--cyan)">{ensemble['spearman_rho_mean']:.3f}</div>
      <div class="metric-label">Mean race-order Spearman</div>
      <div class="kpi-copy">Walk-forward validation across {int(ensemble['Races'])} historical races</div>
    </div>
    <div class="kpi">
      <div class="kpi-value" style="color:var(--green)">{fmt_pct(snapshot['wcc_favorite']['WCC_Prob'])}</div>
      <div class="metric-label">Mercedes WCC probability</div>
      <div class="kpi-copy">{fmt_num(constructor_leader['Points'])} points and +{fmt_num(snapshot['wcc_gap'])} over Ferrari</div>
    </div>
    <div class="kpi">
      <div class="kpi-value" style="color:var(--amber)">{fmt_pct(top_prediction['WinProb'])}</div>
      <div class="metric-label">{html.escape(top_prediction['DriverFull'])} {html.escape(snapshot['forecast_race_name'])} win chance</div>
      <div class="kpi-copy">No driver above 12% win odds in the current diagnostic race-week snapshot</div>
    </div>
  </section>

  <section class="section">
    <div class="wrap grid-2">
      <article class="card">
        <div class="slabel">Championship market</div>
        <div class="ctitle">WDC OUTLOOK</div>
        <p class="copy">Expected final points anchor the bar length; title probability sits on the right. That keeps the visual useful even when the forecast collapses into a two-driver fight.</p>
        <div class="divider"></div>
        {render_wdc_rows(snapshot['wdc'])}
      </article>

      <article class="card">
        <div class="slabel">Live driver table</div>
        <div class="ctitle">DRIVER STANDINGS</div>
        <p class="copy">Current 2026 points through round {snapshot['completed_rounds']}, including sprint points. Last-three finish tokens make momentum legible without burying the audience in raw lap data.</p>
        <div class="divider"></div>
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
          <tbody>
            {render_driver_table(snapshot['driver_standings'])}
          </tbody>
        </table>
      </article>
    </div>
  </section>

  <section class="section">
    <div class="wrap grid-2">
      <article class="card">
        <div class="slabel">Constructor race</div>
        <div class="ctitle">WCC CONTROL ROOM</div>
        <p class="copy">Mercedes already owns the strategic position of the season: {fmt_num(constructor_leader['Points'])} points after three rounds, a +{fmt_num(snapshot['wcc_gap'])} gap to Ferrari, and a current forecast that assigns the team the entire WCC probability mass.</p>
        <div class="divider"></div>
        {render_constructor_rows(snapshot['constructors'])}
      </article>

      <article class="card callout">
        <div class="slabel">2026 regime shift</div>
        <div class="ctitle">CAR ORDER RESET</div>
        <p class="copy">The clearest storyline in the live data is not incremental improvement, but a rapid repricing of team strength under the new regulations. These deltas compare average race points in 2026 against each team's 2025 baseline.</p>
        <div class="divider"></div>
        {render_shift_rows(snapshot['team_shift'])}
      </article>
    </div>
  </section>

  <section class="section">
    <div class="wrap">
      <article class="card">
        <div class="slabel">Diagnostic race-week snapshot</div>
        <div class="ctitle">{html.escape(snapshot['forecast_race_name']).upper()} FORECAST</div>
        <p class="copy">This section turns the model into a presentable race brief: predicted order, uncertainty bounds, podium odds, top-five rates, and failure risk. It is deliberately probability-first, not just a ranked list.</p>
        <div class="forecast-grid">
          {render_prediction_cards(snapshot['sample'])}
        </div>
      </article>
    </div>
  </section>

  <section class="section">
    <div class="wrap grid-3">
      <article class="card">
        <div class="slabel">Signal density</div>
        <div class="ctitle">STRONGEST FEATURE SIGNALS</div>
        <p class="copy">These are the highest absolute correlations against finishing position in the saved feature audit. They are not causal proof or model importance, but they do show where the structure is currently strongest.</p>
        <div class="divider"></div>
        {render_feature_rows(snapshot['feature_corr'])}
      </article>

      <article class="card">
        <div class="slabel">Validation discipline</div>
        <div class="ctitle">HONEST PERFORMANCE</div>
        <p class="copy">Walk-forward validation keeps the model under the same information constraint it faces in production. The current picture is nuanced: the ensemble is best on ranking structure, but the qualifying baseline still wins the simpler winner-pick contest.</p>
        <div class="divider"></div>
        {render_validation_cards(metrics)}
      </article>

      <article class="card">
        <div class="slabel">Model watchlist</div>
        <div class="ctitle">KNOWN GAPS</div>
        <p class="copy">Portfolio-grade analytics should show where the system is strong and where it still has technical debt. These are the three most important issues surfaced during the repo audit.</p>
        <div class="divider"></div>
        {render_watchlist_items()}
      </article>
    </div>
  </section>

  <section class="story">
    <div class="wrap story-grid">
      <div class="story-copy">
        <div class="slabel">Executive narrative</div>
        <div class="ctitle">HOW TO READ THIS SYSTEM</div>
        <p><strong>{html.escape(points_leader['DriverFull'])}</strong> leading the points while <strong>{html.escape(title_favorite['DriverFull'])}</strong> owns the title probability is the exact kind of divergence this project is built to surface. A boardroom audience usually sees only the table; the analytics layer shows what is likely to persist, what is noise, and where current state still understates future control.</p>
        <p><strong>Mercedes</strong> is the early 2026 operating story. The team is up <strong>{snapshot['team_shift'].iloc[0]['PointsShift']:.1f} points per race</strong> versus its 2025 baseline, leads the live constructors table by <strong>{fmt_num(snapshot['wcc_gap'])} points</strong>, and owns a forecasted WCC lock. At the same time, the next-race snapshot shows unusually dispersed winner probabilities, which is exactly why uncertainty needs to be presented explicitly.</p>
        <p>The validation result is also intentionally honest. The <strong>ensemble stack</strong> is stronger than the simple baselines on full-order ranking quality at <strong>{ensemble['spearman_rho_mean']:.3f}</strong> mean Spearman, but it is still behind the qualifying baseline on outright winner hit rate at <strong>{fmt_pct(ensemble['win_accuracy'])}</strong> versus <strong>{fmt_pct(quali['win_accuracy'])}</strong>. That honesty makes the portfolio stronger, not weaker.</p>
      </div>
      <div>
        <div class="slabel">Method stack</div>
        <div class="ctitle">ARCHITECTURE</div>
        <div class="method-grid">
          {methodology_html}
        </div>
      </div>
    </div>
  </section>

  <footer class="footer">
    <div class="wrap footer-grid">
      <div>
        <strong>F1 PREDICTOR | PORTFOLIO DASHBOARD</strong><br/>
        Generated {fmt_date(snapshot['generated_at'])} from live project artifacts<br/>
        Historical span {snapshot['train_year_start']} to {snapshot['train_year_end']} | {snapshot['training_rows']:,} training rows
      </div>
      <div class="footer-note">
        Ensemble weights | XGBoost {weights.get('xgboost', 0):.0%} | Bradley-Terry {weights.get('bradley_terry', 0):.0%} | Monte Carlo {weights.get('monte_carlo', 0):.0%}<br/>
        Walk-forward races | {int(ensemble['Races'])} | Winner accuracy {fmt_pct(ensemble['win_accuracy'])} | Mean MAE {season['mae_positions_mean']:.2f} to {quali['mae_positions_mean']:.2f} across benchmarks
      </div>
    </div>
  </footer>
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
