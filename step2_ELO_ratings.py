"""
STEP 2 — TEAM ELO MODEL (Replaces player-based Delta model)
-----------------------------------------------------------

Pipeline:
    (1) Initialize team ELOs
    (2) Loop through games by date
    (3) Compute expected probabilities using logistic ELO
    (4) Update team ELOs (K-factor)
    (5) Compute Delta = (ELO_home + H) − ELO_away
    (6) Return delta_df + final ELO table
"""

import pandas as pd
import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

ELO_INIT = 1500
K = 20
HOME_ADV = 25   # ELO rating bonus for home team


# ============================================================
# STEP 2 — ELO MODEL
# ============================================================

def run_step2_ELO(player_df, games_df, K=20, HOME_ADV=25):
    print(f"STEP 2 — Running ELO Rating Model (K={K}, HOME_ADV={HOME_ADV})")

    # --------------------------------------------------------
    # Prepare game list in chronological order
    # --------------------------------------------------------
    games = games_df.sort_values("date").copy()

    # Unique team names
    teams = pd.unique(games[["teamname_home", "teamname_away"]].values.ravel())

    # Initialize ELO dictionary
    team_ELO = {team: ELO_INIT for team in teams}

    # Storage for outputs
    rows = []

    # --------------------------------------------------------
    # LOOP THROUGH GAMES
    # --------------------------------------------------------
    for _, g in games.iterrows():

        home = g["teamname_home"]
        away = g["teamname_away"]

        R_home = team_ELO[home]
        R_away = team_ELO[away]

        # Expected home win probability (ELO logistic)
        P_home = 1.0 / (1 + 10 ** ( -((R_home + HOME_ADV) - R_away) / 400 ))

        # Actual outcome
        if g["team_goals_reg_home"] > g["team_goals_reg_away"]:
            S = 1
        elif g["team_goals_reg_home"] == g["team_goals_reg_away"]:
            S = 0.5
        else:
            S = 0

        # ELO update
        delta_home = K * (S - P_home)
        delta_away = -delta_home

        team_ELO[home] += delta_home
        team_ELO[away] += delta_away

        # Delta for Step 3 model
        Delta = (team_ELO[home] + HOME_ADV) - team_ELO[away]

        rows.append({
            "gameId": g["gameId"],
            "date": g["date"],
            "teamname_home": home,
            "teamname_away": away,
            "ELO_home": team_ELO[home],
            "ELO_away": team_ELO[away],
            "Delta": Delta
        })

    delta_df = pd.DataFrame(rows)

    print("STEP 2 COMPLETE — ELO ratings computed.")
    return delta_df, pd.Series(team_ELO, name="ELO")
