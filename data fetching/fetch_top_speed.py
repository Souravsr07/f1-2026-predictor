import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

YEAR = 2026
RACES = ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]

def fetch_speed():

    rows = []

    schedule = fastf1.get_event_schedule(YEAR)
    events = schedule[schedule["EventName"].isin(RACES)]

    for _, event in events.iterrows():

        race = event["EventName"]

        session = fastf1.get_session(YEAR, event["RoundNumber"], "Q")
        session.load()

        for drv in session.drivers:

            laps = session.laps.pick_drivers(drv)

            if laps.empty:
                continue

            lap = laps.pick_fastest()

            if lap is None:
                continue

            tel = lap.get_car_data()

            rows.append({
                "race": race,
                "driver": drv,
                "max_speed": tel["Speed"].max()
            })

    df = pd.DataFrame(rows)

    os.makedirs("data fetching/fetched_data", exist_ok=True)

    df.to_csv(
        "data fetching/fetched_data/top_speed.csv",
        index=False
    )

if __name__ == "__main__":
    fetch_speed()