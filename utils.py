"""
utils.py — shared helpers for BDC 2026 notebooks.

Rink drawing uses sportypy's NHLRink for accurate geometry.  All other
functions are thin wrappers so notebook code stays concise.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sportypy.surfaces.hockey import NHLRink

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Smoothed position columns produced by preprocess.py
X, Y = "x", "y"

#: Colours assigned to teams in order of first appearance
TEAM_COLORS: list[str] = ["#1565C0", "#B71C1C", "#2E7D32", "#E65100"]

# One shared NHLRink instance (stateless — safe to reuse)
_RINK = NHLRink()


# ---------------------------------------------------------------------------
# Rink drawing
# ---------------------------------------------------------------------------


def draw_rink(
    ax: plt.Axes,
    *,
    display_range: str = "full",
    title: str = "",
    title_fontsize: int = 9,
) -> plt.Axes:
    """Draw an NHL rink on *ax* using sportypy.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    display_range:
        Passed straight to ``NHLRink.draw()``.  Useful values:
        ``'full'`` (default), ``'offense'``, ``'defense'``, ``'half'``.
    title:
        Optional axes title.
    """
    _RINK.draw(ax=ax, display_range=display_range)
    ax.set_xlabel("X (feet)", fontsize=8)
    ax.set_ylabel("Y (feet)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    return ax


# ---------------------------------------------------------------------------
# Path plotting
# ---------------------------------------------------------------------------


def plot_player_paths(
    ax: plt.Axes,
    window: pd.DataFrame,
    primary_jersey: str | int,
    event_x: float,
    event_y: float,
    *,
    label: str = "",
    highlight_team: Optional[str] = None,
    x_col: str = X,
    y_col: str = Y,
    display_range: str = "full",
) -> plt.Axes:
    """Plot skater paths from a tracking *window* on a rink.

    Parameters
    ----------
    ax:
        Axes to draw on.
    window:
        Slice of the tracking DataFrame covering the event window.
    primary_jersey:
        Jersey number of the key player (highlighted with a ★ label).
    event_x / event_y:
        Rink coordinates of the triggering event (shown as a gold star).
    label:
        Axes title text.
    highlight_team:
        If set, make *that* team's players bold instead of just the primary.
        Useful for chaser / defender scenarios.
    x_col / y_col:
        Column names for position data.  Defaults to smoothed ``'x'``/``'y'``.
    display_range:
        Passed to :func:`draw_rink`.
    """
    draw_rink(ax, display_range=display_range, title=label)

    teams = sorted(window["Team"].dropna().unique())
    tc = {t: TEAM_COLORS[i % len(TEAM_COLORS)] for i, t in enumerate(teams)}

    for (team, jersey), grp in window.groupby(["Team", "Player Jersey Number"]):
        sort_col = "frame_id" if "frame_id" in grp.columns else "Elapsed_s"
        grp = grp.sort_values(sort_col)
        col = tc.get(team, "grey")
        is_primary = str(jersey) == str(primary_jersey)

        if highlight_team is not None:
            bold = team == highlight_team
        else:
            bold = is_primary

        lw, alpha, zorder = (2.8, 1.0, 6) if bold else (0.9, 0.35, 4)

        xs, ys = grp[x_col].values, grp[y_col].values
        ax.plot(
            xs, ys, color=col, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle="round"
        )

        # Start dot
        ax.scatter(
            xs[0],
            ys[0],
            s=30,
            color=col,
            ec="black",
            lw=0.4,
            zorder=zorder + 1,
            alpha=alpha,
        )

        # Direction arrow at end
        if len(xs) >= 2:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
            if abs(dx) + abs(dy) > 0.5:
                ax.annotate(
                    "",
                    xy=(xs[-1], ys[-1]),
                    xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=lw * 0.8),
                    zorder=zorder + 2,
                )

        # ★ label for primary player
        if is_primary:
            ax.text(
                xs[0] + 1.5,
                ys[0] + 2,
                f"★#{jersey}",
                fontsize=8,
                fontweight="bold",
                color=col,
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, ec=col),
            )

    # Event location
    ax.scatter(
        [event_x],
        [event_y],
        s=250,
        marker="*",
        color="gold",
        ec="black",
        lw=1.2,
        zorder=9,
    )

    # Legend
    patches = [mpatches.Patch(color=tc[t], label=t) for t in teams]
    patches.append(mpatches.Patch(color="gold", label="Event location ★"))
    ax.legend(handles=patches, fontsize=7, loc="upper left", framealpha=0.85)

    return ax


# ---------------------------------------------------------------------------
# Event-window extraction
# ---------------------------------------------------------------------------


def build_tracking_index(
    tracking: pd.DataFrame,
) -> dict[tuple, pd.DataFrame]:
    """Return a dict keyed by (Game, Period_int) → sorted skater DataFrame.

    Assumes *tracking* already has ``Game``, ``Period_int``, and
    ``Elapsed_s`` columns
    """
    skaters = tracking[tracking["Player or Puck"] == "Player"].copy()
    tidx: dict[tuple, pd.DataFrame] = {}
    for (g, p), grp in skaters.groupby(["Game", "Period_int"]):
        tidx[(g, p)] = grp.sort_values("frame_id")
    return tidx


def build_tracking_index_cached(
    tracking: pd.DataFrame,
    cache_path: "str | Path | None" = None,
) -> dict[tuple, pd.DataFrame]:
    """Build (or load from a pickle cache) the tracking index.

    Parameters
    ----------
    tracking:
        DataFrame returned by :func:`load_tracking`.
    cache_path:
        Path to the pickle file (e.g. ``'../data/tracking_index.pkl'``).
        - If the file exists, load from it instead of rebuilding.
        - If the file does not exist, build and save.
        - If ``None``, behaves exactly like :func:`build_tracking_index`.

    Notes
    -----
    Delete the cache file whenever ``tracking.parquet`` is regenerated
    (e.g. after running ``preprocess.py``).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                tidx = pickle.load(fh)
            print(
                f"[cache] Loaded tracking index ({len(tidx)} keys) from {cache_path.name}"
            )
            return tidx

    tidx = build_tracking_index(tracking)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(tidx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1e6
        print(f"[cache] Saved tracking index to {cache_path.name} ({size_mb:.1f} MB)")

    return tidx


def find_examples(
    events: pd.DataFrame,
    tidx: dict[tuple, pd.DataFrame],
    mask: pd.Series,
    *,
    primary_col: str = "Player_Id",
    before_s: float = 3.0,
    after_s: float = 1.0,
    n: int = 4,
    seed: int = 7,
    max_primary_dist_ft: float = 15.0,
) -> list[tuple[pd.Series, pd.DataFrame, str]]:
    """Sample up to *n* event rows matching *mask* and return tracking windows.

    Parameters
    ----------
    max_primary_dist_ft:
        Maximum distance (feet) the primary player may be from the event
        coordinates at the event time.  Events where the identified player
        is farther than this — indicating a bad ``Player_Id`` or a data
        quality issue — are skipped.  Set to ``None`` to disable.

    Returns
    -------
    list of ``(event_row, window_df, primary_jersey_str)``
    """
    results: list[tuple] = []
    sampled = events[mask].sample(frac=1, random_state=seed)

    for _, ev in sampled.iterrows():
        g, p = ev["Game"], ev["Period_int"]
        el, ex, ey = ev["Elapsed_s"], ev["X_Coordinate"], ev["Y_Coordinate"]
        primary = str(ev[primary_col]) if pd.notna(ev.get(primary_col)) else None

        if any(pd.isna(v) for v in [el, ex, ey, p]) or not primary:
            continue
        if (g, p) not in tidx:
            continue

        win = tidx[(g, p)]
        win = win[
            (win["Elapsed_s"] >= el - before_s) & (win["Elapsed_s"] <= el + after_s)
        ]
        if win.empty:
            continue
        if primary not in win["Player Jersey Number"].astype(str).values:
            continue

        # Proximity check: the primary player must pass close to the event
        # location within ±1 s of the event time (tolerates 1-s clock resolution)
        if max_primary_dist_ft is not None and not (pd.isna(ex) or pd.isna(ey)):
            primary_rows = win[
                (win["Player Jersey Number"].astype(str) == primary)
                & (win["Elapsed_s"].between(el - 1, el + 1))
            ]
            if primary_rows.empty:
                continue
            min_dist = np.sqrt(
                (primary_rows[X] - ex) ** 2 + (primary_rows[Y] - ey) ** 2
            ).min()
            if min_dist > max_primary_dist_ft:
                continue

        results.append((ev, win, primary))
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Standard tracking load
# ---------------------------------------------------------------------------


def load_tracking(parquet_path: str) -> pd.DataFrame:
    """Load tracking parquet and add derived columns used throughout notebooks.

    Adds: ``Game``, ``Period_int``, ``Clock_s``, ``Elapsed_s``.
    """
    tracking = pd.read_parquet(parquet_path)

    raw = tracking["Image Id"].str.extract(r"(^\d{4}-\d{2}-\d{2}.*?)_")[0]
    parts = raw.str.extract(r"^(\d{4}-\d{2}-\d{2})\s+(.*?)\s+@\s+(.*)$")
    tracking["Game"] = parts[0] + " " + parts[2] + " @ " + parts[1]
    tracking["Period_int"] = pd.to_numeric(tracking["Period"], errors="coerce")

    # Ensure frame_id exists — baked in by preprocess.py or extracted on the fly
    if "frame_id" not in tracking.columns:
        tracking["frame_id"] = (
            tracking["Image Id"].str.extract(r"_(\d+)$")[0].astype(float)
        )

    def _clk(s: str) -> float:
        try:
            m, c = s.split(":")
            return int(m) * 60 + int(c)
        except Exception:
            return np.nan

    tracking["Clock_s"] = tracking["Game Clock"].apply(_clk)
    tracking["Elapsed_s"] = (tracking["Period_int"] - 1) * 1200 + (
        1200 - tracking["Clock_s"]
    )
    return tracking


def load_events(parquet_path: str) -> pd.DataFrame:
    """Load events parquet and add derived columns."""
    events = pd.read_parquet(parquet_path)
    events["Game"] = (
        events["Date"] + " " + events["Home_Team"] + " @ " + events["Away_Team"]
    )
    events["Period_int"] = pd.to_numeric(events["Period"], errors="coerce")

    def _clk(s: str) -> float:
        try:
            m, c = s.split(":")
            return int(m) * 60 + int(c)
        except Exception:
            return np.nan

    events["Clock_s"] = events["Clock"].apply(_clk)
    events["Elapsed_s"] = (events["Period_int"] - 1) * 1200 + (1200 - events["Clock_s"])
    return events
