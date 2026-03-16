# BDC 2026 — Copilot Instructions

## Project Goal
Build a hockey player stamina/max-effort metric using BDC 2026 tracking + event data. Six scenario notebooks identify moments of likely maximum-speed skating, extract speed profiles, and analyze fatigue patterns.

---

## Data Pipeline

**`preprocess.py`** — 4-step pipeline, all steps accept `force=True` to re-run:
1. `ensure_combined_parquets()` — CSV → parquet (events, shifts, tracking)
2. `impute_positions(force)` — linear interp, `IMPUTE_LIMIT=10`; detects `raw_x` vs `Rink Location X (Feet)` for re-runnability
3. `add_smooth_positions_to_tracking(force)` — SavGol wl=11 poly=2 → columns `x`, `y`
4. `add_speed_to_tracking(force)` — `speed_mph_raw` + `speed_mph_savgol`

**tracking.parquet columns** (post-pipeline):
`Image Id`, `Period`, `Game Clock`, `Player or Puck`, `Team`, `Player Id`, `Player Jersey Number`, `raw_x`, `raw_y`, `Rink Location Z (Feet)`, `Goal Score`, `gap_imputed`, `x`, `y`, `speed_mph_raw`, `speed_mph_savgol`, `frame_id`

**Position columns**: use `x`/`y` (smoothed) for speed/distance calculations, NOT `Rink Location X (Feet)`/`Rink Location Y (Feet)` (raw). `utils.X` and `utils.Y` are `"x"` and `"y"`.

---

## Critical: `frame_id`

- Extracted from `Image Id` suffix (e.g. `2025-10-24 Team A @ Team B_P2_222656` → `frame_id=222656`)
- **Baked permanently into tracking.parquet** — never drop it
- `Elapsed_s` has 1-second resolution (30 frames share the same value) — always sort by `frame_id` within a second
- `load_tracking` extracts `frame_id` on-the-fly if absent (backward compat)
- `build_tracking_index` sorts by `frame_id`
- `plot_player_paths` sorts by `frame_id` (fallback `Elapsed_s`)

---

## `utils.py` — Key Functions

```python
X, Y = "x", "y"   # smoothed position columns

load_tracking(path)          # adds Game/Period_int/Clock_s/Elapsed_s/frame_id
load_events(path)            # adds Game/Period_int/Clock_s/Elapsed_s
build_tracking_index(tracking) -> dict[(Game, Period_int)] -> DataFrame sorted by frame_id
find_examples(events, tidx, mask, *, primary_col, before_s, after_s, n, seed, max_primary_dist_ft=15.0)
plot_player_paths(ax, window, primary_jersey, event_x, event_y, *, label, highlight_team, x_col, y_col, display_range)
draw_rink(ax, *, display_range, title)   # uses sportypy NHLRink
```

**`find_examples` proximity filter**: skips events where the identified primary player is >15 ft from event coords at event time. ~12% of zone entry events have bad `Player_Id`. `Player_Id` IS the carrier (not swapped with `Player_Id_2`).

---

## Notebook Setup Pattern (notebooks 01–06)

```python
import sys, warnings, importlib
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')
import utils
importlib.reload(utils)
from utils import load_tracking, load_events, build_tracking_index, find_examples, plot_player_paths, X, Y
import pandas as pd
import matplotlib.pyplot as plt

tracking = load_tracking('../data/tracking.parquet')
events   = load_events('../data/events.parquet')
shifts   = pd.read_parquet('../data/shifts.parquet')
tidx = build_tracking_index(tracking)
```

**Cell 1 in each notebook is still the OLD inline setup** (raw `pd.read_parquet` + manual derived columns). Do NOT use it — run the utils-based setup cell instead.

---

## Scenario Notebooks (`temp_scenario_exploration/`)

| # | File | Event Filter | Primary | Window | Notes |
|---|------|-------------|---------|--------|-------|
| 01 | `01_carried_zone_entry_carrier.ipynb` | `Zone Entry` + `Detail_1=='Carried'` | `Player_Id` (carrier) | −3s/+1s | Most complete; use as template |
| 02 | `02_carried_zone_entry_chasers.ipynb` | same | carrier (★), focus on defenders | −3s/+1s | Zone-crossing filter; no proximity filter needed |
| 03 | `03_dump_in_out_footrace.ipynb` | `Dump In/Out` | `Player_Id` | −1s/+3s | Post-dump footrace |
| 04 | `04_puck_recovery.ipynb` | `Puck Recovery` | `Player_Id` | −3s/+0s | Approach sprint |
| 05 | `05_takeaway_defender.ipynb` | `Takeaway` | `Player_Id` | −3s/+0s | Closing sprint |
| 06 | `06_shot_shooter.ipynb` | `Shot` | `Player_Id` | −3s/+0s | Shooter approach |

---

## Notebook 02 — Chaser-Specific Logic

**Zone-crossing filter** (in plot cell and analysis cells): defenders who never cross the blue line into the entered zone are excluded — they're "stay-back" defenders, not chasers.

```python
entering_right = ev['X_Coordinate'] >= 0  # blue lines at x=±25
crosses = (grp['x'] > 25).any() if entering_right else (grp['x'] < -25).any()
```

**No proximity filter** applied to chasers (they're supposed to be away from the event location).

**`records` DataFrame columns**: `Game`, `Period_int`, `Player_Id`, `max_speed_mph`, `Elapsed_s`, `elapsed_in_period` (= `1200 - Clock_s`, counts UP from 0).

**`df` depends on `ze_events`, `BEFORE`, `AFTER` defined in the window-timing cell** — run that cell first.

---

## Analysis Patterns

### Window Timing Validation
- Use `frame_id` for sub-second peak offset: `t_rel = (frame_id - event_frame) / FPS`
- For carriers: median peak at −0.8s before event, 68% before, 30% after → −3s/+1s window appropriate
- For chasers: peak timing shifts later (more "after" the blue line crossing)

### Max Speed Distributions (3-panel)
1. Histogram of `max_speed_mph` per event with median + 10th pct lines
2. Observation count histogram: per-period vs per-game (overlaid)
3. Box plots: top 20 players by median max speed (≥3 appearances)

### Fatigue Analysis
**Population-level** (cells 7–8 in nb02): 5-min bin bars by period/game segment — shows trend but confounded by line changes (fresh players replace tired ones, masking fatigue).

**Within-player slopes** (cell 9 in nb02): the right approach —
- `stats.linregress(elapsed_in_period, max_speed_mph)` per player-period
- Slope distribution + t-test vs 0 + "% showing fatigue"
- Appearance-sequence profiles: normalize each player to their own mean, then look at median speed at rank 1, 2, 3... — avoids between-player confound

---

## Player Identity — Critical Join Rule

**`Player_Id` is a jersey number, not a unique player ID.** The same number appears on different teams, so `Player_Id` alone is never a unique key.

| Grain | Correct join key | Wrong |
|-------|-----------------|-------|
| Per game-appearance | `['Game', 'Team', 'Player_Id']` | `['Game', 'Player_Id']` |
| Cross-game (player level) | `['Team', 'Player_Id']` | `['Player_Id']` |

**Apply this rule any time you:**
- `merge` / `join` two DataFrames on player identity
- `groupby` to aggregate per player or per player-game
- Build a lookup dict keyed by player (e.g. `sk_idx`, `ev_idx_by_player`)
- Write results to parquet that downstream notebooks will join against

**Tracking `Team` column** uses short names (`'Home'` / `'Away'`) — map to real team names (parse from `Game` string `"YYYY-MM-DD Home @ Away"`) before using as a join key against shifts or events, which use `'Team A'`, `'Team B'` etc.

**Recommended pattern** — define once at the top of each notebook and reference everywhere:
```python
PLAYER_KEY      = ['Game', 'Team', 'Player_Id']   # per-game grain
PLAYER_TEAM_KEY = ['Team', 'Player_Id']            # cross-game (player-level)
```

---

## Key Lessons

- **`Elapsed_s` is coarse (1s)** — always use `frame_id` for ordering within a period. Sorting by `Elapsed_s` alone gave scrambled paths.
- **`Player_Id` is jersey number** (string), not a database ID — compare with `.astype(str)`.
- **Team column in tracking uses short names**; events uses `Home_Team`/`Away_Team` full names. Match on `Game` string constructed as `Date + Home + @ + Away`.
- **impute_positions must detect column name** after first run (`raw_x` vs `Rink Location X (Feet)`).
- **Observation frequency**: median ~2 carried zone entries per player per game; 63% have ≥2. Per-period is too thin for reliable fatigue metrics — pool across periods or use per-game.
- **Pooling all players for fatigue = confounded** by line changes. Use within-player slopes (linregress per player-period) and appearance-rank profiles (normalize to personal mean).
- **Blue lines at x=±25** in the smoothed coordinate system (after preprocess). Goals at ~x=±89.

## Visualization Preferences
- Use colors from the built in color palettes.
- Use different shapes or line patterns. The visualizations will be printed in black and white, so color alone is not sufficient to distinguish groups.
## Organization Preferences

- output from notebooks should go into an output/ folder, not committed to git