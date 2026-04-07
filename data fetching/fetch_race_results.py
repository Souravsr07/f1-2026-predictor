import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

YEAR = 2026
RACES = ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]

def fetch_race_results():

    rows = []

    schedule = fastf1.get_event_schedule(YEAR)
    events = schedule[schedule["EventName"].isin(RACES)]

    for _, event in events.iterrows():

        race = event["EventName"]

        session = fastf1.get_session(YEAR, event["RoundNumber"], "R")
        session.load()

        results = session.results
        fastest = session.laps.pick_fastest()

        fastest_driver = fastest["Driver"] if fastest is not None else None

        for _, r in results.iterrows():

            rows.append({
                "race": race,
                "driver": r["FullName"],
                "team": r["TeamName"],
                "grid": r["GridPosition"],
                "finish_position": r["Position"],
                "points": r["Points"],
                "dnf": r["Status"] != "Finished",
                "fastest_lap_driver": fastest_driver
            })

    df = pd.DataFrame(rows)

    os.makedirs("data fetching/fetched_data", exist_ok=True)

    df.to_csv(
        "data fetching/fetched_data/race_results.csv",
        index=False
    )

if __name__ == "__main__":
    fetch_race_results()