"""
Normalize and ingest live 2026 CSVs into data/processed/.

Outputs:
  - 2026_live_results.parquet / .csv
  - 2026_live_qualifying.parquet / .csv
  - 2026_live_sprint.parquet / .csv
  - 2026_live_constructor_state.parquet / .csv
  - 2026_live_pace.parquet / .csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, TARGET_YEAR
from utils.name_normalization import (
    build_round_map,
    normalize_driver_code,
    normalize_driver_name,
    normalize_race_name,
    normalize_team_name,
    timedelta_to_seconds,
)


DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent.parent / "data fetching" / "fetched_data"


def _load_csv(source_dir: Path, filename: str) -> pd.DataFrame:
    path = source_dir / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_processed(name: str, df: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_PROCESSED / f"{name}.csv"
    parquet_path = DATA_PROCESSED / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def _normalize_results(df: pd.DataFrame, round_map: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    result["Circuit"] = result["race"].map(normalize_race_name)
    result["Round"] = result["Circuit"].map(round_map)
    result["Year"] = TARGET_YEAR
    result["Driver"] = result["driver"].map(normalize_driver_code)
    result["DriverFull"] = result["driver"].map(normalize_driver_name)
    result["Team"] = result["team"].map(lambda x: normalize_team_name(x, year=TARGET_YEAR))
    result["GridPosition"] = pd.to_numeric(result["grid"], errors="coerce")
    result["FinishPosition"] = pd.to_numeric(result["finish_position"], errors="coerce")
    result["Points"] = pd.to_numeric(result["points"], errors="coerce").fillna(0.0)
    result["DNF"] = result["dnf"].fillna(False).astype(bool)
    result["FastestLapDriver"] = result["fastest_lap_driver"].map(normalize_driver_code)
    result["FastestLap"] = result["Driver"] == result["FastestLapDriver"]
    result["Status"] = result["DNF"].map(lambda x: "DNF" if x else "Finished")

    keep = [
        "Year",
        "Round",
        "Circuit",
        "Driver",
        "DriverFull",
        "Team",
        "GridPosition",
        "FinishPosition",
        "Points",
        "DNF",
        "FastestLap",
        "FastestLapDriver",
        "Status",
    ]
    result = result[keep].dropna(subset=["Circuit", "Driver", "Team"])
    result = result.sort_values(["Round", "FinishPosition", "Driver"]).reset_index(drop=True)
    return result


def _normalize_qualifying(df: pd.DataFrame, round_map: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    result["Circuit"] = result["race"].map(normalize_race_name)
    result["Round"] = result["Circuit"].map(round_map)
    result["Year"] = TARGET_YEAR
    result["Driver"] = result["driver"].map(normalize_driver_code)
    result["DriverFull"] = result["driver"].map(normalize_driver_name)
    result["Team"] = result["team"].map(lambda x: normalize_team_name(x, year=TARGET_YEAR))
    result["QualPosition"] = pd.to_numeric(result["qualifying_position"], errors="coerce")
    result["BestQualTime_s"] = result["best_lap_time"].map(timedelta_to_seconds)
    result["GridPosition"] = result["QualPosition"]

    keep = [
        "Year",
        "Round",
        "Circuit",
        "Driver",
        "DriverFull",
        "Team",
        "QualPosition",
        "GridPosition",
        "BestQualTime_s",
    ]
    result = result[keep].dropna(subset=["Circuit", "Driver", "Team"])
    result = result.sort_values(["Round", "QualPosition", "Driver"]).reset_index(drop=True)
    return result


def _normalize_sprint(df: pd.DataFrame, round_map: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    result["Circuit"] = result["race"].map(normalize_race_name)
    result["Round"] = result["Circuit"].map(round_map)
    result["Year"] = TARGET_YEAR
    result["Driver"] = result["driver"].map(normalize_driver_code)
    result["DriverFull"] = result["driver"].map(normalize_driver_name)
    result["Team"] = result["team"].map(lambda x: normalize_team_name(x, year=TARGET_YEAR))
    result["SprintFinishPosition"] = pd.to_numeric(result["finish_position"], errors="coerce")
    result["SprintPoints"] = pd.to_numeric(result["points"], errors="coerce").fillna(0.0)

    keep = [
        "Year",
        "Round",
        "Circuit",
        "Driver",
        "DriverFull",
        "Team",
        "SprintFinishPosition",
        "SprintPoints",
    ]
    result = result[keep].dropna(subset=["Circuit", "Driver", "Team"])
    result = result.sort_values(["Round", "SprintFinishPosition", "Driver"]).reset_index(drop=True)
    return result


def _normalize_constructor_state(
    results: pd.DataFrame,
    sprint: pd.DataFrame,
    standings_df: pd.DataFrame,
) -> pd.DataFrame:
    if not standings_df.empty:
        state = standings_df.copy()
        state["Team"] = state["team"].map(lambda x: normalize_team_name(x, year=TARGET_YEAR))
        state["Points"] = pd.to_numeric(state["points"], errors="coerce").fillna(0.0)
        state = state[["Team", "Points"]]
    else:
        race_pts = results.groupby("Team", dropna=True)["Points"].sum() if not results.empty else pd.Series(dtype=float)
        sprint_pts = sprint.groupby("Team", dropna=True)["SprintPoints"].sum() if not sprint.empty else pd.Series(dtype=float)
        total = race_pts.add(sprint_pts, fill_value=0.0).reset_index()
        total.columns = ["Team", "Points"]
        state = total

    if state.empty:
        return state

    state = (
        state.groupby("Team", as_index=False)["Points"]
        .sum()
        .sort_values(["Points", "Team"], ascending=[False, True])
        .reset_index(drop=True)
    )
    state["Position"] = state.index + 1
    state["Year"] = TARGET_YEAR
    state["CompletedRounds"] = results["Round"].nunique() if not results.empty else 0
    return state[["Year", "CompletedRounds", "Position", "Team", "Points"]]


def _normalize_pace(
    long_run_df: pd.DataFrame,
    top_speed_df: pd.DataFrame,
    results_df: pd.DataFrame,
    qual_df: pd.DataFrame,
    round_map: dict[str, int],
) -> pd.DataFrame:
    long_run = pd.DataFrame()
    if not long_run_df.empty:
        long_run = long_run_df.copy()
        long_run["Circuit"] = long_run["race"].map(normalize_race_name)
        long_run["Round"] = long_run["Circuit"].map(round_map)
        long_run["Year"] = TARGET_YEAR
        long_run["Team"] = long_run["team"].map(lambda x: normalize_team_name(x, year=TARGET_YEAR))
        long_run["LongRunAvgLap_s"] = long_run["avg_lap_time"].map(timedelta_to_seconds)
        long_run = long_run[["Year", "Round", "Circuit", "Team", "LongRunAvgLap_s"]]
        long_run = (
            long_run.dropna(subset=["Circuit", "Team", "LongRunAvgLap_s"])
            .groupby(["Year", "Round", "Circuit", "Team"], as_index=False)["LongRunAvgLap_s"]
            .mean()
        )
        long_run["LongRunRank"] = (
            long_run.groupby("Circuit")["LongRunAvgLap_s"].rank(method="dense", ascending=True)
        )

    top_speed = pd.DataFrame()
    if not top_speed_df.empty:
        top_speed = top_speed_df.copy()
        top_speed["Circuit"] = top_speed["race"].map(normalize_race_name)
        top_speed["Round"] = top_speed["Circuit"].map(round_map)
        top_speed["Year"] = TARGET_YEAR
        top_speed["Driver"] = top_speed["driver"].map(normalize_driver_code)
        top_speed["TopSpeed_kph"] = pd.to_numeric(top_speed["max_speed"], errors="coerce")

        team_lookup = pd.concat(
            [
                results_df[["Circuit", "Driver", "Team"]],
                qual_df[["Circuit", "Driver", "Team"]],
            ],
            ignore_index=True,
        ).drop_duplicates()
        top_speed = top_speed.merge(team_lookup, on=["Circuit", "Driver"], how="left")
        top_speed = top_speed[["Year", "Round", "Circuit", "Driver", "Team", "TopSpeed_kph"]]
        top_speed = top_speed.dropna(subset=["Circuit", "TopSpeed_kph"])

        top_speed = (
            top_speed.groupby(["Year", "Round", "Circuit", "Team"], as_index=False)
            .agg(
                TopSpeedMean_kph=("TopSpeed_kph", "mean"),
                TopSpeedMax_kph=("TopSpeed_kph", "max"),
            )
        )

    if long_run.empty and top_speed.empty:
        return pd.DataFrame()

    if long_run.empty:
        return top_speed.sort_values(["Round", "TopSpeedMax_kph"], ascending=[True, False]).reset_index(drop=True)
    if top_speed.empty:
        return long_run.sort_values(["Round", "LongRunAvgLap_s"]).reset_index(drop=True)

    pace = long_run.merge(
        top_speed,
        on=["Year", "Round", "Circuit", "Team"],
        how="outer",
    )
    return pace.sort_values(["Round", "Team"]).reset_index(drop=True)


def ingest_live_data(source_dir: Path = DEFAULT_SOURCE_DIR) -> dict[str, pd.DataFrame]:
    round_map = build_round_map()

    raw_results = _load_csv(source_dir, "race_results.csv")
    raw_qual = _load_csv(source_dir, "qualifying_results.csv")
    raw_sprint = _load_csv(source_dir, "sprint_results.csv")
    raw_constructors = _load_csv(source_dir, "constructor_standings.csv")
    raw_long_run = _load_csv(source_dir, "long_run_pace.csv")
    raw_top_speed = _load_csv(source_dir, "top_speed.csv")

    live_results = _normalize_results(raw_results, round_map)
    live_qual = _normalize_qualifying(raw_qual, round_map)
    live_sprint = _normalize_sprint(raw_sprint, round_map)
    live_constructor_state = _normalize_constructor_state(
        live_results,
        live_sprint,
        raw_constructors,
    )
    live_pace = _normalize_pace(
        raw_long_run,
        raw_top_speed,
        live_results,
        live_qual,
        round_map,
    )

    outputs = {
        "2026_live_results": live_results,
        "2026_live_qualifying": live_qual,
        "2026_live_sprint": live_sprint,
        "2026_live_constructor_state": live_constructor_state,
        "2026_live_pace": live_pace,
    }

    for name, frame in outputs.items():
        if not frame.empty:
            _write_processed(name, frame)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and ingest 2026 live CSVs")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing fetched 2026 CSVs",
    )
    args = parser.parse_args()

    outputs = ingest_live_data(args.source_dir)
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()

