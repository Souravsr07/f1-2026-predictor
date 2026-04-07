import pandas as pd

def compute_constructor():

    df = pd.read_csv(
        "data fetching/fetched_data/race_results.csv"
    )

    standings = (
        df.groupby("team")["points"]
        .sum()
        .reset_index()
        .sort_values("points", ascending=False)
    )

    standings.to_csv(
        "data fetching/fetched_data/constructor_standings.csv",
        index=False
    )

if __name__ == "__main__":
    compute_constructor()