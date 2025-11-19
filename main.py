"""
main.py — Runs Step 1, Step 2, and Step 3 (Multinomial Logit + Ordered Probit),
with caching support so Step 2 does not need to be re-run each time.
"""

import os
import pandas as pd

from step1_clean_data import load_and_clean_data
from step2_ELO_ratings import run_step2_ELO
from step3_multinomial_logit import run_step3_multinomial
from step3_ordered_probit import run_step3_ordered_probit


# Objects exposed to Spyder (auto-loaded in Variable Explorer)
player_df = None
games_df = None
delta_df = None
player_FE = None

mlogit_results = None
probit_results = None


# ------------------------------------------------------------
# Helper: load cached Step 1 + Step 2 outputs
# ------------------------------------------------------------

def load_cached_step12():
    global player_df, games_df, delta_df, player_FE

    print("Loading cached Step 1 + Step 2 outputs...")

    player_df = pd.read_csv("data/clean_player.csv", parse_dates=["date"])
    games_df  = pd.read_csv("data/clean_games.csv", parse_dates=["date"])
    delta_df  = pd.read_csv("data/delta_df.csv")

    tmp = pd.read_csv("data/player_FE.csv")
    player_FE = tmp.set_index("playerId")["ANR"]

    print("Loaded cached files.")


# ------------------------------------------------------------
# Helper: save Step 1 + Step 2 outputs
# ------------------------------------------------------------

def save_step12_outputs(player_df, games_df, delta_df, player_FE):
    print("Saving Step 1 + Step 2 outputs to /data ...")

    player_df.to_csv("data/clean_player.csv", index=False)
    games_df.to_csv("data/clean_games.csv", index=False)
    delta_df.to_csv("data/delta_df.csv", index=False)

    out = player_FE.reset_index()
    out.columns = ["playerId", "ANR"]
    out.to_csv("data/player_FE.csv", index=False)

    print("Saved.")


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main(use_cache=True):
    global player_df, games_df, delta_df, player_FE
    global mlogit_results, probit_results

    print("\n=== NHL Forecasting Pipeline ===")

    # ---------- STEP 1 & STEP 2 ----------
    if use_cache and \
       os.path.exists("data/clean_player.csv") and \
       os.path.exists("data/delta_df.csv") and \
       os.path.exists("data/player_FE.csv"):

        load_cached_step12()

    else:
        print("\n=== STEP 1: Loading & Cleaning Data ===")
        player_df, games_df = load_and_clean_data()

        print("\n=== STEP 2: Running Player Model (slow first time) ===")
        delta_df, player_FE = run_step2_ELO(player_df, games_df)

        save_step12_outputs(player_df, games_df, delta_df, player_FE)


    # ---------- STEP 3: MULTINOMIAL LOGIT + ORDERED PROBIT ----------
    print("\n=== STEP 3: Multinomial Logit Forecasting ===")
    mlogit_results = run_step3_multinomial(delta_df, games_df)

    print("\n=== STEP 3.1: Ordered Probit Forecasting ===")
    probit_results = run_step3_ordered_probit(delta_df, games_df)

    print("\n=== DONE — all objects loaded into Variable Explorer ===")


if __name__ == "__main__":
    main(use_cache=False)
