import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

YEAR = 2026
RACES = ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]

def fetch_sprint():

    rows = []

    schedule = fastf1.get_event_schedule(YEAR)
    events = schedule[schedule["EventName"].isin(RACES)]

    for _, event in events.iterrows():

        race = event["EventName"]

        try:

            session = fastf1.get_session(YEAR, event["RoundNumber"], "S")
            session.load()

            results = session.results

            for _, r in results.iterrows():

                rows.append({
                    "race": race,
                    "driver": r["FullName"],
                    "team": r["TeamName"],
                    "finish_position": r["Position"],
                    "points": r["Points"]
                })

        except:
            print("No sprint for", race)

    df = pd.DataFrame(rows)

    os.makedirs("data fetching/fetched_data", exist_ok=True)

    df.to_csv(
        "data fetching/fetched_data/sprint_results.csv",
        index=False
    )

if __name__ == "__main__":
    fetch_sprint()