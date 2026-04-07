import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

YEAR = 2026
RACES = ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]

def fetch_long_runs():

    rows = []

    schedule = fastf1.get_event_schedule(YEAR)
    events = schedule[schedule["EventName"].isin(RACES)]

    for _, event in events.iterrows():

        race = event["EventName"]
        round_no = event["RoundNumber"]

        session = None
        session_name = None

        # try FP2 first
        try:
            session = fastf1.get_session(YEAR, round_no, "FP2")
            session_name = "FP2"
            session.load()
        except:
            pass

        # fallback to FP3
        if session is None:
            try:
                session = fastf1.get_session(YEAR, round_no, "FP3")
                session_name = "FP3"
                session.load()
            except:
                print(f"No practice session found for {race}")
                continue

        print(f"Using {session_name} for {race}")

        laps = session.laps.pick_quicklaps()

        if laps.empty:
            print(f"No usable laps for {race}")
            continue

        # group by driver + stint
        stints = laps.groupby(["Driver","Stint"])

        for (driver, stint), stint_laps in stints:

            if len(stint_laps) < 5:
                continue

            avg_time = stint_laps["LapTime"].mean()

            rows.append({
                "race": race,
                "driver": driver,
                "team": stint_laps["Team"].iloc[0],
                "stint": stint,
                "laps": len(stint_laps),
                "avg_lap_time": avg_time
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No long runs detected")
        return

    # team averages
    team_pace = (
        df.groupby(["race","team"])["avg_lap_time"]
        .mean()
        .reset_index()
    )

    os.makedirs("data fetching/fetched_data", exist_ok=True)

    team_pace.to_csv(
        "data fetching/fetched_data/long_run_pace.csv",
        index=False
    )

    print("Long run pace dataset saved")


if __name__ == "__main__":
    fetch_long_runs()