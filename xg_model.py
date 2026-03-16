"""
xg_model.py — in-house expected goals (xG) model.

Fits a logistic regression on shots from the events DataFrame.
Features: shot distance to goal, shot angle, and shot type (Detail_1).
Blocked shots are excluded — the goalie never faced them.

Coordinate system (same as events.parquet X_Coordinate / Y_Coordinate):
  - Centre ice at (0, 0)
  - Blue lines at x ≈ ±25 ft
  - Goal mouths at x ≈ ±89 ft, y = 0

Usage
-----
    import sys; sys.path.insert(0, '..')
    from xg_model import fit_xg_model, sum_xg_for_entries

    xg_pipe = fit_xg_model(events)
    obs_with_xg = sum_xg_for_entries(obs_raw, events, xg_pipe)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GOAL_X = 89.0  # goal mouth x-coordinate (absolute value), ft from centre ice
SHOT_WINDOW_S = 20.0  # seconds after zone entry to accumulate xG

# Unblocked shot events that count toward xG
_XG_EVENTS = {"Shot", "Goal", "Missed Shot"}


def _shot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shot_dist_ft, shot_angle_deg, shot_type, is_goal to a shot rows DataFrame."""
    df = df.copy()
    x = df["X_Coordinate"].astype(float)
    y = df["Y_Coordinate"].astype(float)

    # Determine which goal is being attacked based on shot x-coordinate.
    # Shots with x > 0 are taken in the right-hand zone, aimed at (GOAL_X, 0).
    goal_x = np.where(x >= 0, GOAL_X, -GOAL_X)

    df["shot_dist_ft"] = np.sqrt((x - goal_x) ** 2 + y**2)

    # Angle from the goal-mouth centre line.
    # 0° = straight on, 90° = shot from directly to the side.
    dx = np.abs(goal_x - x)
    dy = np.abs(y)
    df["shot_angle_deg"] = np.degrees(np.arctan2(dy, dx))

    df["shot_type"] = df["Detail_1"].fillna("Unknown").astype(str)
    df["is_goal"] = (df["Event"] == "Goal").astype(int)
    return df


def fit_xg_model(events_df: pd.DataFrame, verbose: bool = True) -> Pipeline:
    """
    Train an xG logistic regression on all unblocked shots in events_df.

    Parameters
    ----------
    events_df : events DataFrame produced by utils.load_events()
    verbose   : print a summary if True

    Returns
    -------
    Fitted sklearn Pipeline (ColumnTransformer + LogisticRegression).
    """
    shots = events_df[events_df["Event"].isin(_XG_EVENTS)].copy()
    shots = _shot_features(shots)
    shots = shots.dropna(subset=["shot_dist_ft", "shot_angle_deg"])

    X = shots[["shot_dist_ft", "shot_angle_deg", "shot_type"]]
    y = shots["is_goal"]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["shot_dist_ft", "shot_angle_deg"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["shot_type"],
            ),
        ]
    )
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=1000))])
    pipe.fit(X, y)

    if verbose:
        shots["xg"] = pipe.predict_proba(X)[:, 1]
        print(f"xG model trained on {len(shots):,} unblocked shots")
        print(f"  Goals      : {y.sum()}  ({y.mean():.1%} conversion)")
        print(f"  Mean xG    : {shots['xg'].mean():.4f}")
        print(f"  Shot types : {shots['shot_type'].value_counts().to_dict()}")
        coef_num = pipe.named_steps["lr"].coef_[0]
        print(f"  Coef (dist, angle): {coef_num[:2].round(3)}")

    return pipe


def score_shots(pipe: Pipeline, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of all unblocked shot-type event rows with an 'xg' column.
    Rows with missing coordinates are dropped.
    """
    shots = events_df[events_df["Event"].isin(_XG_EVENTS)].copy()
    shots = _shot_features(shots)
    valid = shots.dropna(subset=["shot_dist_ft", "shot_angle_deg"]).copy()
    valid["xg"] = pipe.predict_proba(
        valid[["shot_dist_ft", "shot_angle_deg", "shot_type"]]
    )[:, 1]
    return valid


def sum_xg_for_entries(
    obs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pipe: Pipeline,
    window_s: float = SHOT_WINDOW_S,
) -> pd.DataFrame:
    """
    For each zone-entry observation, sum xG of all unblocked shots in the
    next `window_s` seconds (same game & period).

    Adds three columns to a copy of obs_df:
      sum_xg         — total expected goals in the window (0 if no shots)
      max_xg         — highest single-shot xG in the window (0 if no shots)
      n_shots_window — number of unblocked shots in the window

    Parameters
    ----------
    obs_df    : observation DataFrame from drs_builder.build_observations()
    events_df : events DataFrame from utils.load_events()
    pipe      : fitted Pipeline from fit_xg_model()
    window_s  : seconds after zone entry to scan for shots
    """
    scored = score_shots(pipe, events_df)

    sum_xg_list, max_xg_list, n_shots_list = [], [], []
    for _, row in obs_df.iterrows():
        window = scored[
            (scored["Game"] == row["Game"])
            & (scored["Period_int"] == row["Period_int"])
            & (scored["Elapsed_s"] > row["Elapsed_s"])
            & (scored["Elapsed_s"] <= row["Elapsed_s"] + window_s)
        ]
        sum_xg_list.append(window["xg"].sum())
        max_xg_list.append(window["xg"].max() if len(window) > 0 else 0.0)
        n_shots_list.append(len(window))

    out = obs_df.copy()
    out["sum_xg"] = sum_xg_list
    out["max_xg"] = max_xg_list
    out["n_shots_window"] = n_shots_list
    return out


# ---------------------------------------------------------------------------
# Saved-model I/O (produced by xg_model.ipynb tuning workflow)
# ---------------------------------------------------------------------------


def load_xg_model(path: str | Path) -> tuple:
    """
    Load a saved xG model from disk.

    Parameters
    ----------
    path : path to the .pkl file saved by xg_model.ipynb (joblib format).

    Returns
    -------
    (pipe, feature_cols)
        pipe         — fitted sklearn Pipeline (or CalibratedClassifierCV)
        feature_cols — list of column names expected by pipe.predict_proba()

    Example
    -------
        xg_pipe, feat_cols = xgm.load_xg_model('data/xg_model_best.pkl')
        shots_df = xgm.score_shots_v2(xg_pipe, feat_cols, events)
    """
    import joblib

    bundle = joblib.load(path)
    pipe = bundle["pipe"]
    feats = bundle["features"]
    print(f"Loaded xG model from {path}")
    print(f"  Model  : {bundle.get('best_model_name', 'unknown')}")
    print(f"  Test AUC: {bundle.get('test_auc', float('nan')):.4f}")
    print(f"  OT excluded: {bundle.get('ot_excluded', False)}")
    print(f"  Features: {feats}")
    return pipe, feats


def score_shots_v2(pipe, feature_cols: list, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all unblocked shots using a pipeline loaded from disk.

    Uses the full feature set produced by xg_model.ipynb's engineer_features(),
    which may include distance², angle×distance, is_slot, period, clock_s, etc.

    Parameters
    ----------
    pipe         : fitted pipeline (from load_xg_model)
    feature_cols : list of feature columns (from load_xg_model)
    events_df    : events DataFrame from utils.load_events()

    Returns
    -------
    DataFrame of shot rows with an added 'xg' column.
    """
    shots = events_df[events_df["Event"].isin(_XG_EVENTS)].copy()

    x = shots["X_Coordinate"].astype(float)
    y = shots["Y_Coordinate"].astype(float)
    goal_x = np.where(x >= 0, GOAL_X, -GOAL_X)

    shots["shot_dist_ft"] = np.sqrt((x - goal_x) ** 2 + y**2)
    shots["shot_angle_deg"] = np.degrees(np.arctan2(np.abs(y), np.abs(goal_x - x)))
    shots["shot_dist_sq"] = shots["shot_dist_ft"] ** 2
    shots["dist_x_angle"] = shots["shot_dist_ft"] * shots["shot_angle_deg"]
    shots["is_sharp_angle"] = (shots["shot_angle_deg"] > 55).astype(int)
    shots["is_slot"] = (
        (shots["shot_dist_ft"] < 30) & (shots["shot_angle_deg"] < 45)
    ).astype(int)
    shots["shot_type"] = shots["Detail_1"].fillna("Unknown").astype(str)
    shots["period"] = shots["Period_int"].astype(int)
    shots["clock_s"] = (
        shots.get("Clock_s", pd.Series(600, index=shots.index))
        .fillna(600)
        .astype(float)
    )
    shots["is_overtime"] = (shots["period"] > 3).astype(int)
    shots["is_goal"] = (shots["Event"] == "Goal").astype(int)

    # ── Strength state ───────────────────────────────────────────────────────
    home_sk = pd.to_numeric(
        shots.get("Home_Team_Skaters", pd.Series(5, index=shots.index)), errors="coerce"
    ).fillna(5)
    away_sk = pd.to_numeric(
        shots.get("Away_Team_Skaters", pd.Series(5, index=shots.index)), errors="coerce"
    ).fillna(5)
    is_home = shots["Team"] == shots["Home_Team"]
    shots["shooter_skaters"] = np.where(is_home, home_sk, away_sk).astype(int)
    shots["defender_skaters"] = np.where(is_home, away_sk, home_sk).astype(int)
    shots["skater_advantage"] = shots["shooter_skaters"] - shots["defender_skaters"]
    shots["is_power_play"] = (shots["skater_advantage"] > 0).astype(int)
    shots["is_penalty_kill"] = (shots["skater_advantage"] < 0).astype(int)
    shots["is_empty_net"] = (shots["defender_skaters"] >= 6).astype(int)
    shots["is_3on3"] = (
        (shots["shooter_skaters"] == 3) & (shots["defender_skaters"] == 3)
    ).astype(int)

    def _strength_label(sh, de):
        if sh == 5 and de == 5:
            return "5v5"
        if sh == 3 and de == 3:
            return "3v3"
        if de >= 6:
            return "EN"  # defending team pulled goalie
        if sh > de:
            return "PP"  # 5v4, 4v3, etc.
        if sh < de:
            return "PK"  # 4v5, 3v4, etc.
        return "other"

    shots["strength_state"] = [
        _strength_label(s, d)
        for s, d in zip(shots["shooter_skaters"], shots["defender_skaters"])
    ]

    # Exclude overtime — model was trained on regulation only
    shots = shots[shots["is_overtime"] == 0].copy()

    valid = shots.dropna(subset=["shot_dist_ft", "shot_angle_deg"]).copy()
    # Keep only feature cols that are present (graceful degradation)
    available = [c for c in feature_cols if c in valid.columns]
    valid["xg"] = pipe.predict_proba(valid[available])[:, 1]
    return valid


def sum_xg_for_entries_v2(
    obs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pipe,
    feature_cols: list,
    window_s: float = SHOT_WINDOW_S,
) -> pd.DataFrame:
    """
    Like sum_xg_for_entries but uses the richer feature set from score_shots_v2.
    Drop-in replacement when using a model loaded from xg_model_best.pkl.
    """
    scored = score_shots_v2(pipe, feature_cols, events_df)

    sum_xg_list, max_xg_list, n_shots_list = [], [], []
    for _, row in obs_df.iterrows():
        window = scored[
            (scored["Game"] == row["Game"])
            & (scored["Period_int"] == row["Period_int"])
            & (scored["Elapsed_s"] > row["Elapsed_s"])
            & (scored["Elapsed_s"] <= row["Elapsed_s"] + window_s)
        ]
        sum_xg_list.append(window["xg"].sum())
        max_xg_list.append(window["xg"].max() if len(window) > 0 else 0.0)
        n_shots_list.append(len(window))

    out = obs_df.copy()
    out["sum_xg"] = sum_xg_list
    out["max_xg"] = max_xg_list
    out["n_shots_window"] = n_shots_list
    return out
