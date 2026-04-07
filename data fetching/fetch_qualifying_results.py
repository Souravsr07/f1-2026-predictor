import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

YEAR = 2026
RACES = ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]

def fetch_qualifying():

    rows = []

    schedule = fastf1.get_event_schedule(YEAR)
    events = schedule[schedule["EventName"].isin(RACES)]

    for _, event in events.iterrows():

        race = event["EventName"]

        session = fastf1.get_session(YEAR, event["RoundNumber"], "Q")
        session.load()

        results = session.results

        for _, r in results.iterrows():

            best = r["Q3"] if pd.notna(r["Q3"]) else r["Q2"] if pd.notna(r["Q2"]) else r["Q1"]

            rows.append({
                "race": race,
                "driver": r["FullName"],
                "team": r["TeamName"],
                "qualifying_position": r["Position"],
                "best_lap_time": best
            })

    df = pd.DataFrame(rows)

    os.makedirs("data fetching/fetched_data", exist_ok=True)

    df.to_csv(
        "data fetching/fetched_data/qualifying_results.csv",
        index=False
    )

if __name__ == "__main__":
    fetch_qualifying()