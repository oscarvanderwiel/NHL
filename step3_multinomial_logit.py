"""
STEP 3 - MULTINOMIAL LOGIT OUTCOME MODEL

Inputs:
    delta_df  - from Step 2 (gameId, Delta, R_home, R_away)
    games_df  - contains true match outcomes:
                team_goals_reg_home, team_goals_reg_away

Output:
    model_fit - fitted MNLogit model
    pred_df   - predicted probabilities:
                P_home, P_tie, P_away
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


# ============================================================
# 1. ENCODE MATCH OUTCOME
# ============================================================

def encode_match_result(df):
    out = []
    for h_score, a_score in zip(df['team_goals_reg_home'], df['team_goals_reg_away']):
        if h_score > a_score:
            out.append(2)
        elif h_score == a_score:
            out.append(1)
        else:
            out.append(0)
    return np.array(out)


# ============================================================
# 2. FIT MULTINOMIAL LOGIT
# ============================================================

def fit_multinomial_logit(df):
    y = encode_match_result(df)
    X = sm.add_constant(df["Delta"])
    model = sm.MNLogit(y, X)
    result = model.fit(method="newton", maxiter=200, disp=False)
    return result


# ============================================================
# 3. PREDICT PROBABILITIES
# ============================================================

def predict_mnlogit(result, df, tie_threshold=0):
    X = sm.add_constant(df["Delta"])
    pred = result.predict(X)

    out = df.copy()
    out["P_away"] = pred.iloc[:, 0]
    out["P_tie"]  = pred.iloc[:, 1]
    out["P_home"] = pred.iloc[:, 2]

    # ---- METHOD A: Tie-band classification ----
    diff = abs(out["P_home"] - out["P_away"])

    out["pred_class"] = np.where(
        diff < tie_threshold,      # if probabilities close
        1,                         # classify as tie
        np.where(out["P_home"] > out["P_away"], 2, 0)   # else home=2, away=0
    )

    return out[["gameId", "Delta", "P_home", "P_tie", "P_away", "pred_class"]]




# ============================================================
# WRAPPER FUNCTION
# ============================================================

def run_step3_multinomial(delta_df, games_df):
    print("STEP 3 - Multinomial Logit Model")

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

    model_fit = fit_multinomial_logit(df)
    pred_df = predict_mnlogit(model_fit, df)

    # Add actual_class to predictions
    pred_df["actual_class"] = df["actual_class"].values

    print("STEP 3 COMPLETE - Multinomial probabilities computed.")
    return model_fit, pred_df



