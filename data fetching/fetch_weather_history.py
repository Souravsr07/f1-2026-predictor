import argparse
from pathlib import Path
import sys

import fastf1
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_PROCESSED
from utils.name_normalization import normalize_race_name


fastf1.Cache.enable_cache("cache")


def fetch_weather_history(years: list[int], session_types: list[str]) -> pd.DataFrame:
    rows = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            round_no = int(event["RoundNumber"])
            race = event["EventName"]

            for session_type in session_types:
                try:
                    session = fastf1.get_session(year, round_no, session_type)
                    session.load(laps=False, telemetry=False, weather=True, messages=False)
                except Exception as exc:
                    print(f"Skipping {year} {race} {session_type}: {exc}")
                    continue

                weather = session.weather_data
                if weather is None or weather.empty:
                    continue

                rows.append(
                    {
                        "year": year,
                        "round": round_no,
                        "race": race,
                        "session": session_type,
                        "air_temp_mean": round(float(weather["AirTemp"].mean()), 2) if "AirTemp" in weather.columns else None,
                        "track_temp_mean": round(float(weather["TrackTemp"].mean()), 2) if "TrackTemp" in weather.columns else None,
                        "humidity_mean": round(float(weather["Humidity"].mean()), 2) if "Humidity" in weather.columns else None,
                        "wind_speed_mean": round(float(weather["WindSpeed"].mean()), 2) if "WindSpeed" in weather.columns else None,
                        "rainfall_any": bool(weather["Rainfall"].any()) if "Rainfall" in weather.columns else False,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical weather from FastF1")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2027)))
    parser.add_argument("--sessions", nargs="+", default=["Q", "R"])
    args = parser.parse_args()

    df = fetch_weather_history(args.years, args.sessions)
    if not df.empty:
        df["Circuit"] = df["race"].map(normalize_race_name)
    out_dir = Path("data fetching/fetched_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weather_history.csv"
    df.to_csv(out_path, index=False)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    processed_csv = DATA_PROCESSED / "weather_history.csv"
    processed_parquet = DATA_PROCESSED / "weather_history.parquet"
    df.to_csv(processed_csv, index=False)
    df.to_parquet(processed_parquet, index=False)
    print(f"Saved {len(df):,} weather rows -> {out_path}")
    print(f"Processed copies -> {processed_csv} and {processed_parquet}")


if __name__ == "__main__":
    main()
