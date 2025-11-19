"""
01_clean_data.py

STEP 1 — CLEAN & STRUCTURE THE DATA
-----------------------------------

Prepares the player- and game-level data for the player-rating model.
This revised version is aligned with the simplified Step 2 design:

(1) Load player & game data
(2) Basic cleaning and type conversions
(3) Remove non–on-ice observations
(4) Assign position groups (Forward, Defense, Goalie)
(5) Compute goalie save rate (only needed goalie metric)
(6) Merge team names, identify opponents
(7) Return clean DataFrames for Step 2
"""

import pandas as pd
import numpy as np


def load_and_clean_data(player_path="playergamedata.csv",
                        games_path="gamedata.csv"):
    """
    Load and clean raw player/game data.

    Returns:
        player_df (DataFrame)
        games_df (DataFrame)
    """

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    player = pd.read_csv(player_path)
    games  = pd.read_csv(games_path)

    # ---------------------------------------------------------
    # 2. BASIC CLEANING
    # ---------------------------------------------------------
    player["gameId"]   = player["gameId"].astype(str)
    player["playerId"] = player["playerId"].astype(str)
    games["gameId"]    = games["gameId"].astype(str)

    player["date"] = pd.to_datetime(player["date"])
    games["date"]  = pd.to_datetime(games["date"])

    # Ensure numeric types
    numeric_cols = ["toi_seconds", "plusMinus", 
                    "goals", "assists", "shots",
                    "team_goals_reg", "home_goals_reg", "away_goals_reg"]
    for col in numeric_cols:
        if col in player.columns:
            player[col] = pd.to_numeric(player[col], errors="coerce")

    games["team_goals_reg_home"] = pd.to_numeric(games["team_goals_reg_home"], errors="coerce")
    games["team_goals_reg_away"] = pd.to_numeric(games["team_goals_reg_away"], errors="coerce")

    # Remove players with no ice time
    player["toi_seconds"] = pd.to_numeric(player["toi_seconds"], errors="coerce")
    player = player[player["toi_seconds"] > 0]

    # Standardize home indicator
    player["home"] = player["home"].astype(int)

    # ---------------------------------------------------------
    # 3. POSITION GROUPING
    # ---------------------------------------------------------
    def map_pos(p):
        if p in ["C", "L", "R"]:
            return "Forward"
        elif p == "D":
            return "Defense"
        elif p == "G":
            return "Goalie"
        else:
            return np.nan

    player["pos_group"] = player["position"].astype(str).str.strip().apply(map_pos)
    player = player.dropna(subset=["pos_group"])

    # ---------------------------------------------------------
    # 4. ATTACH TEAM NAMES + OPPONENT NAMES
    # ---------------------------------------------------------
    player = player.merge(
        games[["gameId", "teamname_home", "teamname_away"]],
        on="gameId",
        how="left"
    )

    player["teamname"] = np.where(player["home"] == 1,
                                  player["teamname_home"],
                                  player["teamname_away"])

    player["opp_team"] = np.where(player["home"] == 1,
                                  player["teamname_away"],
                                  player["teamname_home"])

    # ---------------------------------------------------------
    # 5. GOALIE SAVE RATE (ONLY PORTION WE ACTUALLY NEED)
    # ---------------------------------------------------------
    # Shots by opponent: sum skaters' shots
    opp_shots = (
        player[player["pos_group"] != "Goalie"]
        .groupby(["gameId", "teamname"])["shots"]
        .sum()
        .reset_index()
        .rename(columns={"shots": "team_shots"})
    )

    player = player.merge(
        opp_shots.rename(columns={"teamname": "opp_teamname"}),
        left_on=["gameId", "opp_team"],
        right_on=["gameId", "opp_teamname"],
        how="left"
    )

    # shots_against for goalies
    player["shots_against"] = np.where(
        player["pos_group"] == "Goalie",
        player["team_shots"],
        np.nan
    )

    # goals_against
    player["goals_against"] = np.where(
        player["pos_group"] == "Goalie",
        np.where(player["home"] == 1,
                 player["away_goals_reg"],
                 player["home_goals_reg"]),
        np.nan
    )

    # save rate
    player["save_rate"] = np.where(
        player["pos_group"] == "Goalie",
        np.where(player["shots_against"] > 0,
                 1 - player["goals_against"] / player["shots_against"],
                 np.nan),
        np.nan
    )

    # ---------------------------------------------------------
    # 6. RETURN CLEAN DATA
    # ---------------------------------------------------------
    print("\nSTEP 1 COMPLETE — Data cleaned & structured.")
    print(f"Players: {len(player):,}   Games: {len(games):,}")

    return player, games


if __name__ == "__main__":
    p, g = load_and_clean_data()
    print(p.head())
    print(g.head())
