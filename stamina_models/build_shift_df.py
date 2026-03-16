"""
build_shift_df.py
=================
Shared preprocessing for all stamina model notebooks.

Usage
-----
    from build_shift_df import build_shift_df, PREDICTORS, POS_COLORS
    shift_df = build_shift_df('../data')          # loads cache if present
    shift_df = build_shift_df('../data', force=True)  # always recomputes

Output columns
--------------
Shift identity:
    Game, Team, Period_int, Player_Id, shift_number
    shift_start_s, shift_end_s, shift_duration_s
    rest_time_s, is_first_shift

Tracking-derived:
    n_frames, ice_min
    dist_ft, dist_ft_per_min
    mean_abs_x, std_x, mean_abs_y, std_y, frac_deep_zone, speed_mean

Fatigue proxy:
    cumulative_ice_min_before

Position:
    position  ('C', 'F', or 'D' — from hierarchical GMM in 11_position_inference.ipynb)
              use C(position, Treatment('D')) in formulas; D is the reference level

Game state:
    Home_Team_Goals, Away_Team_Goals, score_diff

Acceleration (per-minute rates + raw counts for offset models):
    accelerations_per_min          — any threshold crossing (accel + decel) / ice_min
    accel_count                    — raw integer count behind accelerations_per_min
    acceleration_events_per_min    — sustained bouts (>= SUSTAINED_FRAMES) / ice_min
    accel_events_count             — raw integer count behind acceleration_events_per_min
    mean_accel_magnitude           — mean(|accel|) across all shift frames (mph/s)

The cache is saved as shift_df.parquet in the same directory as this file.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))
warnings.filterwarnings("ignore")

import utils  # noqa: E402
from utils import load_tracking, X, Y  # noqa: E402

# ── Public constants ──────────────────────────────────────────────────────────
FPS = 30.0
DIST_CONV = 5280 / 3600 / FPS  # ft per frame per mph
MIN_FRAMES = 300  # ~10 s of coverage
ACCEL_THRESH = 2.0  # mph/s — acceleration/deceleration threshold
SUSTAINED_FRAMES = 5  # min consecutive frames for a "sustained" event

POS_COLORS = {"C": "#2E7D32", "F": "#1565C0", "D": "#B71C1C"}
POS_LABELS = {"C": "Centers", "F": "Forwards", "D": "Defensemen"}

# Shared predictor formula string — import this in every model notebook
# D is the Treatment reference so C and F coefficients are vs. defensemen.
PREDICTORS = (
    "rest_time_min + "
    "is_first_int + "
    "C(position, Treatment('D')) + "
    "cumulative_ice_min_before + "
    "score_diff + "
    "shift_duration_min + "
    "C(Period_int)"
)

_CACHE_PATH = _HERE / "shift_df.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_clock(clock_str: str) -> float:
    """'MM:SS' countdown clock → elapsed seconds within period (0 = period start)."""
    parts = str(clock_str).strip().split(":")
    return 1200 - (int(parts[0]) * 60 + int(parts[1]))


def count_sustained_bouts(signal: np.ndarray, threshold: float, min_frames: int) -> int:
    """Count contiguous runs where signal > threshold lasting >= min_frames frames."""
    above = (signal > threshold).astype(np.int8)
    padded = np.concatenate([[0], above, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int(np.sum((ends - starts) >= min_frames))


# ── Main entry point ──────────────────────────────────────────────────────────


def build_shift_df(
    data_dir: str | Path = "../data", force: bool = False
) -> pd.DataFrame:
    """
    Build (or load from cache) the per-shift feature DataFrame.

    Parameters
    ----------
    data_dir : path to the directory containing tracking.parquet, events.parquet,
               shifts.parquet, player_clusters.parquet, position_model.pkl
    force    : if True, ignore the cache and recompute from scratch

    Returns
    -------
    pd.DataFrame with all columns documented in the module docstring.
    """
    if not force and _CACHE_PATH.exists():
        print(f"Loading cached shift_df from {_CACHE_PATH}")
        return pd.read_parquet(_CACHE_PATH)

    data_dir = Path(data_dir)
    print("Building shift_df from scratch …")

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    tracking = load_tracking(str(data_dir / "tracking.parquet"))
    events = pd.read_parquet(data_dir / "events.parquet")

    print(f"  Tracking rows : {len(tracking):,}")
    print(f"  Speed present : {'speed_mph_savgol' in tracking.columns}")

    # ── Step 2: Clean shifts ──────────────────────────────────────────────────
    shifts_raw = pd.read_parquet(data_dir / "shifts.parquet")

    shifts_sk = shifts_raw[
        (shifts_raw["Player_Id"].astype(str) != "Go")
        & shifts_raw["Player_Id"].astype(str).str.match(r"^\d+$")
        & shifts_raw["period"].isin([1, 2, 3])
    ].copy()

    shifts_sk["Game"] = (
        shifts_sk["Date"]
        + " "
        + shifts_sk["Home_Team"]
        + " @ "
        + shifts_sk["Away_Team"]
    )
    shifts_sk["Period_int"] = shifts_sk["period"].astype(int)
    shifts_sk["Player_Id"] = shifts_sk["Player_Id"].astype(str)
    shifts_sk["shift_start_s"] = shifts_sk["start_clock"].apply(parse_clock)
    shifts_sk["shift_end_s"] = shifts_sk["end_clock"].apply(parse_clock)

    shifts_sk = shifts_sk[
        (shifts_sk["shift_end_s"] > shifts_sk["shift_start_s"])
        & (shifts_sk["shift_start_s"] >= 0)
        & (shifts_sk["shift_end_s"] <= 1200)
    ].copy()

    print(f"  Skater shifts (P1–P3): {len(shifts_sk):,}")

    # ── Step 3: Rest time & first-shift flag ──────────────────────────────────
    shifts_sk = shifts_sk.sort_values(
        ["Game", "Team", "Period_int", "Player_Id", "shift_start_s"]
    ).copy()

    rest_times, is_first_lst = [], []
    for (_, _, _, _), grp in shifts_sk.groupby(
        ["Game", "Team", "Period_int", "Player_Id"], sort=False
    ):
        grp = grp.sort_values("shift_start_s")
        prev_end = None
        for _, row in grp.iterrows():
            if prev_end is None:
                rest_times.append(0.0)
                is_first_lst.append(True)
            else:
                rest_times.append(max(0.0, row["shift_start_s"] - prev_end))
                is_first_lst.append(False)
            prev_end = row["shift_end_s"]

    shifts_sk["rest_time_s"] = rest_times
    shifts_sk["is_first_shift"] = is_first_lst

    # ── Step 4: Build skater tracking index ───────────────────────────────────
    sk = tracking[tracking["Player or Puck"] == "Player"].copy()
    sk = sk[sk["Player Jersey Number"].astype(str).str.match(r"^\d+$")]
    sk = sk.sort_values(["Game", "Period_int", "Player Jersey Number", "frame_id"])

    if "elapsed_in_period_s" not in sk.columns:
        sk["elapsed_in_period_s"] = sk["Elapsed_s"] - (sk["Period_int"] - 1) * 1200

    sk_idx: dict = {}
    for (game, period, pid), grp in sk.groupby(
        ["Game", "Period_int", "Player Jersey Number"], sort=False
    ):
        sk_idx[(game, int(period), str(pid))] = grp.sort_values("frame_id")

    has_accel_col = "accel_mph_s" in sk.columns
    print(
        f"  Tracking groups: {len(sk_idx):,}  |  accel_mph_s present: {has_accel_col}"
    )

    # ── Step 5: Per-shift distance & position features ─────────────────────────
    shift_rows = []
    for _, sh in shifts_sk.iterrows():
        key = (sh["Game"], sh["Period_int"], sh["Player_Id"])
        tr = sk_idx.get(key)
        if tr is None or tr.empty:
            continue

        window = tr[
            (tr["elapsed_in_period_s"] >= sh["shift_start_s"])
            & (tr["elapsed_in_period_s"] <= sh["shift_end_s"])
        ]
        n_frames = len(window)
        if n_frames < MIN_FRAMES:
            continue

        spd = window["speed_mph_savgol"].to_numpy(dtype=float)
        ice_min = n_frames / FPS / 60
        dist_ft = float(np.nansum(spd) * DIST_CONV)
        xs = window[X].to_numpy(dtype=float)
        ys = window[Y].to_numpy(dtype=float)
        abs_x = np.abs(xs)

        # ── Acceleration metrics ──────────────────────────────────────────────
        if has_accel_col:
            accel = window["accel_mph_s"].to_numpy(dtype=float)
        else:
            accel = np.gradient(spd) * FPS

        above = (accel > ACCEL_THRESH).astype(np.int8)
        below = (accel < -ACCEL_THRESH).astype(np.int8)
        n_accel = int(np.sum(np.diff(above) > 0))
        n_decel = int(np.sum(np.diff(below) > 0))
        raw_count = n_accel + n_decel

        n_sust_accel = count_sustained_bouts(accel, ACCEL_THRESH, SUSTAINED_FRAMES)
        n_sust_decel = count_sustained_bouts(-accel, ACCEL_THRESH, SUSTAINED_FRAMES)
        events_count = n_sust_accel + n_sust_decel

        shift_rows.append(
            {
                # identity
                "Game": sh["Game"],
                "Team": sh["Team"],
                "Period_int": sh["Period_int"],
                "Player_Id": sh["Player_Id"],
                "shift_number": sh["shift_number"],
                "shift_start_s": sh["shift_start_s"],
                "shift_end_s": sh["shift_end_s"],
                "shift_duration_s": sh["shift_end_s"] - sh["shift_start_s"],
                "rest_time_s": sh["rest_time_s"],
                "is_first_shift": sh["is_first_shift"],
                # tracking
                "n_frames": n_frames,
                "ice_min": ice_min,
                "dist_ft": dist_ft,
                "dist_ft_per_min": dist_ft / ice_min,
                # position features
                "mean_abs_x": float(np.nanmean(abs_x)),
                "std_x": float(np.nanstd(xs)),
                "mean_abs_y": float(np.nanmean(np.abs(ys))),
                "std_y": float(np.nanstd(ys)),
                "frac_deep_zone": float(np.mean(abs_x > 60)),
                "speed_mean": float(np.nanmean(spd)),
                # acceleration
                "accel_count": raw_count,
                "accelerations_per_min": raw_count / ice_min,
                "accel_events_count": events_count,
                "acceleration_events_per_min": events_count / ice_min,
                "mean_accel_magnitude": float(np.nanmean(np.abs(accel))),
            }
        )

    shift_df = pd.DataFrame(shift_rows)

    # ── Step 6: Cumulative ice time before this shift ──────────────────────────
    shift_df = shift_df.sort_values(
        ["Game", "Team", "Player_Id", "Period_int", "shift_start_s"]
    ).reset_index(drop=True)
    shift_df["cumulative_ice_min_before"] = shift_df.groupby(
        ["Game", "Team", "Player_Id"]
    )["ice_min"].transform(lambda s: s.cumsum().shift(1, fill_value=0.0))

    print(f"  Shifts with >={MIN_FRAMES} frames: {len(shift_df):,}")

    # ── Step 7: Position labels ───────────────────────────────────────────────
    # Uses position_labels.parquet from 11_position_inference.ipynb (C/F/D).
    # Join key must include Team — Player_Id is a jersey number only and can
    # collide across teams playing in the same game.
    pos_path = data_dir / "position_labels.parquet"

    if pos_path.exists():
        player_pos = pd.read_parquet(
            pos_path
        )  # columns: Game, Team, Player_Id, position, conf
        shift_df = shift_df.merge(
            player_pos[["Game", "Team", "Player_Id", "position"]],
            on=["Game", "Team", "Player_Id"],
            how="left",
        )
        n_ok = shift_df["position"].notna().sum()
        print(
            f"  Position labels joined: {n_ok:,}/{len(shift_df):,} ({100*n_ok/len(shift_df):.1f}%)"
        )
        # Fill unknown with 'F' as conservative fallback (centers + wingers both
        # skate more like forwards than defensemen for fatigue purposes)
        shift_df["position"] = shift_df["position"].fillna("F")
    else:
        print(
            "  WARNING: position_labels.parquet not found — run 11_position_inference.ipynb first."
        )
        print(
            "  Falling back to heuristic: players above team median mean_abs_x = F, else D."
        )
        player_med = (
            shift_df.groupby(["Team", "Player_Id"])["mean_abs_x"]
            .median()
            .rename("player_med_x")
        )
        shift_df = shift_df.join(player_med, on=["Team", "Player_Id"])
        # shift_df["position"] = np.where(
        #     shift_df["mean_abs_x"] >= shift_df["player_med_x"], "F", "D"
        # )

    # ── Step 8: Score differential via merge_asof ─────────────────────────────
    ev_sc = events.copy()
    ev_sc["Game"] = (
        ev_sc["Date"] + " " + ev_sc["Home_Team"] + " @ " + ev_sc["Away_Team"]
    )
    ev_sc["Period_int"] = ev_sc["Period"].astype(int)
    ev_sc["elapsed_s"] = ev_sc["Clock"].apply(parse_clock)
    ev_sc["game_elapsed_s"] = (ev_sc["Period_int"] - 1) * 1200 + ev_sc["elapsed_s"]
    ev_sc = (
        ev_sc[
            [
                "Game",
                "game_elapsed_s",
                "Home_Team",
                "Home_Team_Goals",
                "Away_Team_Goals",
            ]
        ]
        .sort_values(["Game", "game_elapsed_s"])
        .reset_index(drop=True)
    )

    # 'Game' may have been renamed to 'Game_x' by the position merge if both sides
    # carried a 'Game' column — detect and handle
    game_col = "Game" if "Game" in shift_df.columns else "Game_x"

    shift_df["game_elapsed_s"] = (shift_df["Period_int"] - 1) * 1200 + shift_df[
        "shift_start_s"
    ]

    pieces = []
    for game, grp in shift_df.groupby(game_col, sort=False):
        ev_game = ev_sc[ev_sc["Game"] == game].sort_values("game_elapsed_s")
        grp_sorted = grp.sort_values("game_elapsed_s")
        m = pd.merge_asof(
            grp_sorted, ev_game, on="game_elapsed_s", direction="backward"
        )
        pieces.append(m)

    shift_df = pd.concat(pieces, ignore_index=True)
    # merge_asof creates Game_x / Game_y when both sides carry 'Game' — restore the column
    if "Game_x" in shift_df.columns:
        shift_df = shift_df.rename(columns={"Game_x": "Game"}).drop(
            columns=["Game_y"], errors="ignore"
        )
    shift_df["Home_Team_Goals"] = shift_df["Home_Team_Goals"].fillna(0).astype(int)
    shift_df["Away_Team_Goals"] = shift_df["Away_Team_Goals"].fillna(0).astype(int)
    shift_df["score_diff"] = np.where(
        shift_df["Team"] == shift_df["Home_Team"],
        shift_df["Home_Team_Goals"] - shift_df["Away_Team_Goals"],
        shift_df["Away_Team_Goals"] - shift_df["Home_Team_Goals"],
    )
    shift_df = shift_df.drop(columns=["game_elapsed_s", "Home_Team"], errors="ignore")

    # ── Save cache ────────────────────────────────────────────────────────────
    shift_df.to_parquet(_CACHE_PATH, index=False)
    print(f"  Saved cache → {_CACHE_PATH}")
    print(f"  Final shape : {shift_df.shape}")

    return shift_df
