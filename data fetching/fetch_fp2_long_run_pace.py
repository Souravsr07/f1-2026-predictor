import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

YEAR = 2026
RACES = ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]

def fetch_fp2():

    rows = []

    schedule = fastf1.get_event_schedule(YEAR)
    events = schedule[schedule["EventName"].isin(RACES)]

    for _, event in events.iterrows():

        race = event["EventName"]

        session = fastf1.get_session(YEAR, event["RoundNumber"], "FP2")
        session.load()

        laps = session.laps.pick_quicklaps()

        pace = laps.groupby("Team")["LapTime"].mean()

        for team, lap in pace.items():

            rows.append({
                "race": race,
                "team": team,
                "avg_lap_time": lap
            })

    df = pd.DataFrame(rows)

    os.makedirs("data fetching/fetched_data", exist_ok=True)

    df.to_csv(
        "data fetching/fetched_data/fp2_long_run_pace.csv",
        index=False
    )

if __name__ == "__main__":
    fetch_fp2()