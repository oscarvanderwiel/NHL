"""
STEP 3.1 - ORDERED PROBIT MODEL

Inputs:
    delta_df  - from Step 2 (gameId, Delta)
    games_df  - contains true match outcomes

Output:
    result    - fitted Ordered Probit model
    pred_df   - predicted probabilities:
                P_home, P_tie, P_away
"""

import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ============================================================
# 1. ENCODE OUTCOME
# ============================================================

def encode_ordered_outcome(df):
    y = []
    for h_score, a_score in zip(df['team_goals_reg_home'], df['team_goals_reg_away']):
        if h_score > a_score:
            y.append(1)
        elif h_score == a_score:
            y.append(0)
        else:
            y.append(-1)
    return np.array(y)




# ============================================================
# 2. FIT ORDERED PROBIT
# ============================================================

def fit_ordered_probit(df):
    y = encode_ordered_outcome(df)
    X = df[["Delta"]]

    model = OrderedModel(y, X, distr="probit")
    result = model.fit(method="bfgs", maxiter=200, disp=False)
    return result


# ============================================================
# 3. PREDICT PROBABILITIES
# ============================================================

def predict_ordered_probit(result, df, tie_threshold=0):
    X = df[["Delta"]]
    pred = result.predict(X)

    out = df.copy()
    out["P_away"] = pred.iloc[:, 0]
    out["P_tie"]  = pred.iloc[:, 1]
    out["P_home"] = pred.iloc[:, 2]

    # ---- METHOD A: Tie-band classification ----
    diff = abs(out["P_home"] - out["P_away"])

    out["pred_class"] = np.where(
        diff < tie_threshold,      # if probabilities close
        1,                         # tie
        np.where(out["P_home"] > out["P_away"], 2, 0)
    )

    return out[["gameId", "Delta", "P_home", "P_tie", "P_away", "pred_class"]]




# ============================================================
# WRAPPER FUNCTION
# ============================================================

def run_step3_ordered_probit(delta_df, games_df):
    print("STEP 3.1 - Ordered Probit Model")

    df = delta_df.merge(
        games_df[["gameId", "team_goals_reg_home", "team_goals_reg_away"]],
        on="gameId",
        how="left"
    )

    # Encode actual match result: home=2, tie=1, away=0
    df["actual_class"] = np.where(
        df["team_goals_reg_home"] > df["team_goals_reg_away"], 2,
        np.where(
            df["team_goals_reg_home"] == df["team_goals_reg_away"], 1, 0
        )
    )

    result = fit_ordered_probit(df)
    pred_df = predict_ordered_probit(result, df)

    # Add actual result to predictions
    pred_df["actual_class"] = df["actual_class"].values

    print("STEP 3.1 COMPLETE - Ordered Probit probabilities computed.")
    return result, pred_df


