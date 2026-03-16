from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from combine_csvs import combine_all


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

EVENTS_PARQUET = DATA_DIR / "events.parquet"
SHIFTS_PARQUET = DATA_DIR / "shifts.parquet"
TRACKING_PARQUET = DATA_DIR / "tracking.parquet"
SKIPPED_SEGMENTS_CSV = DATA_DIR / "skipped_segments.csv"

IMPUTE_LIMIT = (
    10  # max consecutive NaN frames to linearly interpolate (≈0.33 s at 30 Hz)
)


def ensure_combined_parquets() -> None:
    required = [EVENTS_PARQUET, SHIFTS_PARQUET, TRACKING_PARQUET]
    missing = [path for path in required if not path.exists()]

    if not missing:
        print(
            "Step 1/5: Combined parquet files already exist. Skipping CSV combine step."
        )
        return

    print("Step 1/5: Missing parquet files detected.")
    for path in missing:
        print(f"  - {path.name}")

    print("Combining CSV files into parquet...")
    combine_all(data_dir=DATA_DIR, output_dir=DATA_DIR)


def _savgol_xy_segment(values: pd.Series) -> np.ndarray:
    """Apply SavGol(wl=11, poly=2) to a single continuous segment of X or Y positions."""
    arr = values.to_numpy(dtype=float)
    # savgol can't handle NaN — return raw if any are present
    if np.any(np.isnan(arr)):
        return arr
    n = len(arr)
    if n < 5:
        return arr
    # window must be odd and <= n
    window = min(11, n if n % 2 == 1 else n - 1)
    if window < 5:
        return arr
    return savgol_filter(arr, window_length=window, polyorder=2, mode="interp")


def _savgol_deriv_segment(values: pd.Series, fps: float = 30.0) -> np.ndarray:
    """Analytical first derivative of the SavGol polynomial on a single contiguous
    segment.  Returns velocity in **ft/s** (same coordinate units as raw_x/raw_y).

    Uses the same window/polyorder as the position smoother so the derivative is
    consistent with the smoothed curve — no extra finite-difference step needed.
    Short or NaN-containing segments return all-NaN (they were not smoothed either).
    """
    arr = values.to_numpy(dtype=float)
    n = len(arr)
    if np.any(np.isnan(arr)) or n < 5:
        return np.full(n, np.nan)
    # Match the window selection logic in _savgol_xy_segment
    window = min(11, n if n % 2 == 1 else n - 1)
    if window < 5:
        return np.full(n, np.nan)
    dt = 1.0 / fps  # seconds per frame
    return savgol_filter(arr, window_length=window, polyorder=2, deriv=1, delta=dt)


def _impute_linear(s: pd.Series, limit: int = IMPUTE_LIMIT) -> pd.Series:
    """Linear interpolation capped at `limit` frames. Short-series safe."""
    safe_limit = min(limit, max(1, len(s) - 1))
    return s.interpolate(method="linear", limit=safe_limit, limit_direction="both")


def impute_positions(force: bool = False) -> None:
    """Fill short NaN gaps in skater X/Y via linear interpolation.

    Operates on the raw coordinate columns before renaming or smoothing.
    Adds a ``gap_imputed`` boolean column to flag imputed frames.
    Gaps longer than IMPUTE_LIMIT frames are left as NaN — SavGol will
    skip those segments and log them to skipped_segments.csv.
    """
    if not TRACKING_PARQUET.exists():
        raise FileNotFoundError(f"Tracking parquet not found: {TRACKING_PARQUET}")

    print(
        f"Step 2/5: Imputing short NaN gaps in skater positions (limit={IMPUTE_LIMIT} frames)..."
    )
    tracking = pd.read_parquet(TRACKING_PARQUET)

    if not force and "gap_imputed" in tracking.columns:
        print("  Column 'gap_imputed' already present. Skipping imputation step.")
        return

    # Support re-running after add_smooth_positions_to_tracking has already renamed columns
    if "raw_x" in tracking.columns:
        x_col, y_col = "raw_x", "raw_y"
    else:
        x_col, y_col = "Rink Location X (Feet)", "Rink Location Y (Feet)"

    tracking["frame_id"] = tracking["Image Id"].str.extract(r"_(\d+)$")[0].astype(float)
    tracking["_game"] = tracking["Image Id"].str.extract(r"(^\d{4}-\d{2}-\d{2}.*?)_")[0]
    tracking = tracking.sort_values(
        ["_game", "Period", "Player Id", "frame_id"]
    ).reset_index(drop=True)

    was_nan = tracking[x_col].isna()
    out_x = tracking[x_col].copy()
    out_y = tracking[y_col].copy()

    skater_mask = tracking["Player or Puck"] == "Player"
    for _, grp in tracking[skater_mask].groupby(["_game", "Period", "Player Id"]):
        idx = grp.index
        out_x[idx] = _impute_linear(grp[x_col])
        out_y[idx] = _impute_linear(grp[y_col])

    tracking[x_col] = out_x
    tracking[y_col] = out_y
    tracking["gap_imputed"] = was_nan & out_x.notna()

    tracking = tracking.drop(columns=["_game"], errors="ignore")  # keep frame_id
    tracking.to_parquet(TRACKING_PARQUET, index=False, engine="pyarrow")

    filled = int(tracking["gap_imputed"].sum())
    remain = int(out_x[skater_mask].isna().sum())
    print(
        f"  ✓ Gaps filled: {filled:,} frames  |  Still NaN: {remain:,} skater frames (gaps > {IMPUTE_LIMIT})"
    )


def add_speed_to_tracking(force: bool = False) -> None:
    """Compute speed columns from already-smoothed positions.

    Requires ``add_smooth_positions_to_tracking()`` to have run first so that
    ``raw_x``/``raw_y`` and ``x``/``y`` columns exist.

    - ``speed_mph_raw``    — frame-to-frame speed from raw sensor positions
    - ``speed_mph_savgol`` — frame-to-frame speed from SavGol-smoothed positions
      (naturally smooth; no second SavGol pass needed)
    """
    if not TRACKING_PARQUET.exists():
        raise FileNotFoundError(f"Tracking parquet not found: {TRACKING_PARQUET}")

    print("Step 4/5: Adding speed columns to tracking data (if needed)...")
    tracking = pd.read_parquet(TRACKING_PARQUET)

    if (
        not force
        and "speed_mph_savgol" in tracking.columns
        and tracking["speed_mph_savgol"].notna().any()
    ):
        print("  Column 'speed_mph_savgol' already populated. Skipping speed step.")
        return

    if "x" not in tracking.columns:
        raise RuntimeError(
            "Smoothed position columns ('x'/'y') not found. "
            "Run add_smooth_positions_to_tracking() first."
        )

    if "frame_id" not in tracking.columns:
        tracking["frame_id"] = (
            tracking["Image Id"].str.extract(r"_(\d+)$")[0].astype(float)
        )
    tracking["_game"] = tracking["Image Id"].str.extract(r"(^\d{4}-\d{2}-\d{2}.*?)_")[0]

    tracking = tracking.sort_values(
        ["_game", "Period", "Player Id", "frame_id"]
    ).reset_index(drop=True)

    tracking["speed_mph_raw"] = np.nan
    tracking["speed_mph_savgol"] = np.nan

    FPS = 30.0
    FT_PER_S_TO_MPH = 0.681818  # 1 ft/s = 0.681818 mph

    players = tracking[tracking["Player or Puck"] == "Player"].copy()

    grp_keys = ["_game", "Period", "Player Id"]
    players["_img_prev"] = players.groupby(grp_keys)["frame_id"].shift(1)
    players["_frame_diff"] = players["frame_id"] - players["_img_prev"]
    contiguous = players["_frame_diff"] == 1

    # ── Raw speed: chord differences on sensor positions ──────────────────────
    def _chord_speed(x_col, y_col):
        dx = players[x_col] - players.groupby(grp_keys)[x_col].shift(1)
        dy = players[y_col] - players.groupby(grp_keys)[y_col].shift(1)
        # ft/frame × 30 frames/s × 0.681818 mph/(ft/s)
        mph = np.sqrt(dx**2 + dy**2) * FPS * FT_PER_S_TO_MPH
        mph = np.where(contiguous, mph, np.nan)
        return np.where((mph < 0) | (mph > 30), np.nan, mph)

    players["speed_mph_raw"] = _chord_speed("raw_x", "raw_y")

    # ── Savgol speed: analytical derivative of the smoothed curve ─────────────
    # Segment boundaries: any gap > 1 frame starts a new segment.
    players["_segment_id"] = (
        players.groupby(grp_keys)["_frame_diff"]
        .transform(lambda s: (s != 1).cumsum())
        .astype(int)
    )
    players["speed_mph_savgol"] = np.nan

    seg_keys = grp_keys + ["_segment_id"]
    for _, seg in players.groupby(seg_keys):
        # _savgol_deriv_segment returns ft/s (same units as raw_x/raw_y per second)
        vx = _savgol_deriv_segment(seg["x"], fps=FPS)  # ft/s
        vy = _savgol_deriv_segment(seg["y"], fps=FPS)  # ft/s
        spd = np.sqrt(vx**2 + vy**2) * FT_PER_S_TO_MPH  # mph
        spd = np.where(spd > 30, np.nan, spd)  # clamp implausible values
        players.loc[seg.index, "speed_mph_savgol"] = spd

    tracking.loc[players.index, "speed_mph_raw"] = players["speed_mph_raw"]
    tracking.loc[players.index, "speed_mph_savgol"] = players["speed_mph_savgol"]

    tracking = tracking.drop(columns=["_game"], errors="ignore")  # keep frame_id
    tracking.to_parquet(TRACKING_PARQUET, index=False, engine="pyarrow")

    valid_raw = int(tracking["speed_mph_raw"].notna().sum())
    valid_sg = int(tracking["speed_mph_savgol"].notna().sum())
    print(f"  ✓ tracking.parquet updated with speed columns")
    print(f"    speed_mph_raw:    {valid_raw:,} non-null rows (from raw_x/raw_y)")
    print(f"    speed_mph_savgol: {valid_sg:,} non-null rows (from smoothed x/y)")


def add_accel_to_tracking(force: bool = False) -> None:
    """Compute signed acceleration from speed_mph_savgol via SavGol derivative.

    - ``accel_mph_s`` — d(speed_mph_savgol)/dt in **mph/s** (signed).
      Positive = player is speeding up; negative = slowing down.
      NaN at segment boundaries and short/gappy segments.
    """
    if not TRACKING_PARQUET.exists():
        raise FileNotFoundError(f"Tracking parquet not found: {TRACKING_PARQUET}")

    print("Step 5/5: Adding acceleration column to tracking data (if needed)...")
    tracking = pd.read_parquet(TRACKING_PARQUET)

    if (
        not force
        and "accel_mph_s" in tracking.columns
        and tracking["accel_mph_s"].notna().any()
    ):
        print("  Column 'accel_mph_s' already populated. Skipping acceleration step.")
        return

    if "speed_mph_savgol" not in tracking.columns:
        raise RuntimeError("Run add_speed_to_tracking() first.")

    if "frame_id" not in tracking.columns:
        tracking["frame_id"] = (
            tracking["Image Id"].str.extract(r"_(\d+)$")[0].astype(float)
        )
    tracking["_game"] = tracking["Image Id"].str.extract(r"(^\d{4}-\d{2}-\d{2}.*?)_")[0]
    tracking = tracking.sort_values(
        ["_game", "Period", "Player Id", "frame_id"]
    ).reset_index(drop=True)

    tracking["accel_mph_s"] = np.nan
    FPS = 30.0

    players = tracking[tracking["Player or Puck"] == "Player"].copy()
    grp_keys = ["_game", "Period", "Player Id"]
    players["_img_prev"] = players.groupby(grp_keys)["frame_id"].shift(1)
    players["_frame_diff"] = players["frame_id"] - players["_img_prev"]
    players["_segment_id"] = (
        players.groupby(grp_keys)["_frame_diff"]
        .transform(lambda s: (s != 1).cumsum())
        .astype(int)
    )

    seg_keys = grp_keys + ["_segment_id"]
    for _, seg in players.groupby(seg_keys):
        spd = seg["speed_mph_savgol"].to_numpy(dtype=float)
        idx = seg.index.to_numpy()
        n = len(spd)

        # Split further on NaN-speed frames so we compute accel on each
        # contiguous run of valid speeds — never skipping a whole segment.
        valid_mask = ~np.isnan(spd)
        # Label contiguous runs of True in valid_mask
        run_id = np.cumsum(
            np.concatenate([[False], np.diff(valid_mask.astype(int)) != 0])
        )
        for rid in np.unique(run_id[valid_mask]):
            sub_pos = np.where((run_id == rid) & valid_mask)[0]
            if len(sub_pos) < 5:
                continue
            sub_spd = spd[sub_pos]
            sub_idx = idx[sub_pos]
            wl = min(11, len(sub_spd) if len(sub_spd) % 2 == 1 else len(sub_spd) - 1)
            if wl < 5:
                continue
            # SavGol derivative of speed → mph/s
            accel = savgol_filter(
                sub_spd, window_length=wl, polyorder=2, deriv=1, delta=1.0 / FPS
            )
            players.loc[sub_idx, "accel_mph_s"] = accel

    tracking.loc[players.index, "accel_mph_s"] = players["accel_mph_s"]
    tracking = tracking.drop(columns=["_game"], errors="ignore")
    tracking.to_parquet(TRACKING_PARQUET, index=False, engine="pyarrow")

    valid = int(tracking["accel_mph_s"].notna().sum())
    print(f"  ✓ tracking.parquet updated with accel_mph_s column")
    print(f"    accel_mph_s: {valid:,} non-null rows (mph/s, signed)")


def add_smooth_positions_to_tracking(force: bool = False) -> None:
    """Rename raw X/Y columns to raw_x/raw_y and add SavGol-smoothed x/y columns."""
    if not TRACKING_PARQUET.exists():
        raise FileNotFoundError(f"Tracking parquet not found: {TRACKING_PARQUET}")

    print("Step 3/5: Adding smoothed X/Y positions to tracking data (if needed)...")
    tracking = pd.read_parquet(TRACKING_PARQUET)

    if not force and "x" in tracking.columns and tracking["x"].notna().any():
        print("Columns 'x'/'y' already populated. Skipping position smoothing step.")
        return

    # Rename raw coordinate columns
    rename_map = {}
    if "Rink Location X (Feet)" in tracking.columns:
        rename_map["Rink Location X (Feet)"] = "raw_x"
    if "Rink Location Y (Feet)" in tracking.columns:
        rename_map["Rink Location Y (Feet)"] = "raw_y"
    if rename_map:
        tracking = tracking.rename(columns=rename_map)

    if "frame_id" not in tracking.columns:
        tracking["frame_id"] = (
            tracking["Image Id"].str.extract(r"_(\d+)$")[0].astype(float)
        )
    tracking["_game"] = tracking["Image Id"].str.extract(r"(^\d{4}-\d{2}-\d{2}.*?)_")[0]

    tracking = tracking.sort_values(
        ["_game", "Period", "Player Id", "frame_id"]
    ).reset_index(drop=True)

    # Default smoothed = raw (puck rows stay unchanged)
    tracking["x"] = tracking["raw_x"].copy()
    tracking["y"] = tracking["raw_y"].copy()

    players = tracking[tracking["Player or Puck"] == "Player"].copy()

    players["_image_prev"] = players.groupby(["_game", "Period", "Player Id"])[
        "frame_id"
    ].shift(1)
    players["_frame_diff"] = players["frame_id"] - players["_image_prev"]

    players["_segment_id"] = (
        players.groupby(["_game", "Period", "Player Id"])["_frame_diff"]
        .transform(lambda s: (s != 1).cumsum())
        .astype(int)
    )

    group_keys = ["_game", "Period", "Player Id", "_segment_id"]
    skipped_log = []

    for (g, per, pid, seg), grp in players.groupby(group_keys):
        if grp["raw_x"].isna().any() or grp["raw_y"].isna().any():
            # Segment still has NaN after imputation — skip SavGol, keep raw, log it
            skipped_log.append(
                {
                    "game": g,
                    "period": per,
                    "player_id": pid,
                    "segment_id": int(seg),
                    "n_frames": len(grp),
                    "n_nan_x": int(grp["raw_x"].isna().sum()),
                    "n_nan_y": int(grp["raw_y"].isna().sum()),
                }
            )
            continue  # x/y already initialised to raw_x/raw_y above

        players.loc[grp.index, "x"] = _savgol_xy_segment(grp["raw_x"])
        players.loc[grp.index, "y"] = _savgol_xy_segment(grp["raw_y"])

    tracking.loc[players.index, "x"] = players["x"]
    tracking.loc[players.index, "y"] = players["y"]

    # Write skipped-segment log
    if skipped_log:
        skipped_df = pd.DataFrame(skipped_log)
        skipped_df.to_csv(SKIPPED_SEGMENTS_CSV, index=False)
        skipped_frames = sum(d["n_frames"] for d in skipped_log)
        print(
            f"  - Skipped {len(skipped_log)} segments ({skipped_frames:,} frames) — logged to {SKIPPED_SEGMENTS_CSV.name}"
        )
    else:
        print("  - No segments skipped (all NaN gaps were imputed)")

    tracking = tracking.drop(columns=["_game"], errors="ignore")  # keep frame_id
    tracking.to_parquet(TRACKING_PARQUET, index=False, engine="pyarrow")

    valid = int(tracking["x"].notna().sum())
    print(f"  ✓ tracking.parquet updated with smoothed position columns")
    print(f"    raw_x / raw_y: original sensor positions (renamed)")
    print(f"    x / y:         SavGol(wl=11, poly=2) smoothed positions")
    print(f"    x non-null rows: {valid:,}")


def main(force: bool = True) -> None:
    ensure_combined_parquets()
    impute_positions(force=force)
    add_smooth_positions_to_tracking(force=force)
    add_speed_to_tracking(force=force)
    add_accel_to_tracking(force=force)
    print("Preprocessing complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Skip steps that already have output (default: re-run all)",
    )
    parser.set_defaults(force=True)
    args = parser.parse_args()
    main(force=args.force)
