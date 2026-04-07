import argparse
from pathlib import Path

import fastf1
import pandas as pd


fastf1.Cache.enable_cache("cache")


def fetch_starting_grid(year: int, races: list[str] | None = None) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if races:
        schedule = schedule[schedule["EventName"].isin(races)]

    rows = []
    for _, event in schedule.iterrows():
        round_no = int(event["RoundNumber"])
        race = event["EventName"]

        try:
            quali = fastf1.get_session(year, round_no, "Q")
            quali.load()
            race_session = fastf1.get_session(year, round_no, "R")
            race_session.load()
        except Exception as exc:
            print(f"Skipping {year} {race}: {exc}")
            continue

        qual_results = quali.results
        race_results = race_session.results
        if qual_results is None or race_results is None:
            continue

        qual_df = pd.DataFrame(
            {
                "driver": qual_results["FullName"],
                "team": qual_results["TeamName"],
                "qualifying_position": qual_results["Position"],
            }
        )
        grid_df = pd.DataFrame(
            {
                "driver": race_results["FullName"],
                "team": race_results["TeamName"],
                "grid_position": race_results["GridPosition"],
            }
        )

        merged = qual_df.merge(grid_df, on=["driver", "team"], how="outer")
        merged["race"] = race
        merged["year"] = year
        merged["round"] = round_no
        merged["grid_penalty_places"] = (
            pd.to_numeric(merged["grid_position"], errors="coerce")
            - pd.to_numeric(merged["qualifying_position"], errors="coerce")
        )
        merged["penalty_flag"] = merged["grid_penalty_places"].fillna(0).ne(0)
        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch final starting grids from FastF1")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--races", nargs="*", default=None)
    args = parser.parse_args()

    df = fetch_starting_grid(args.year, args.races)
    out_dir = Path("data fetching/fetched_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "starting_grid.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} starting-grid rows -> {out_path}")


if __name__ == "__main__":
    main()
