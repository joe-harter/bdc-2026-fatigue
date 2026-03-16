import glob
from pathlib import Path

import pandas as pd


def combine_events(data_dir: Path, output_dir: Path) -> Path:
    events_files = sorted(glob.glob(str(data_dir / "*Events.csv")))
    if not events_files:
        raise FileNotFoundError(f"No Events CSV files found in {data_dir}")

    print(f"Processing {len(events_files)} Events files...")
    events_dfs = [
        pd.read_csv(file, dtype={"Player_Id": str, "Player_Id_2": str})
        for file in events_files
    ]
    events_combined = pd.concat(events_dfs, ignore_index=True)
    output_path = output_dir / "events.parquet"
    events_combined.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"✓ events.parquet created: {len(events_combined)} rows")
    return output_path


def combine_shifts(data_dir: Path, output_dir: Path) -> Path:
    shifts_files = sorted(glob.glob(str(data_dir / "*Shifts.csv")))
    if not shifts_files:
        raise FileNotFoundError(f"No Shifts CSV files found in {data_dir}")

    print(f"Processing {len(shifts_files)} Shifts files...")
    shifts_dfs = [pd.read_csv(file, dtype={"Player_Id": str}) for file in shifts_files]
    shifts_combined = pd.concat(shifts_dfs, ignore_index=True)
    output_path = output_dir / "shifts.parquet"
    shifts_combined.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"✓ shifts.parquet created: {len(shifts_combined)} rows")
    return output_path


def combine_tracking(data_dir: Path, output_dir: Path) -> Path:
    tracking_files = sorted(glob.glob(str(data_dir / "*Tracking_P*.csv")))
    if not tracking_files:
        raise FileNotFoundError(f"No Tracking CSV files found in {data_dir}")

    print(f"Processing {len(tracking_files)} Tracking files...")
    tracking_dfs = []
    for file in tracking_files:
        df = pd.read_csv(
            file,
            dtype={
                "Player Id": str,
                "Period": str,
                "Player Jersey Number": str,
                "Goal Score": str,
            },
        )
        tracking_dfs.append(df)

    tracking_combined = pd.concat(tracking_dfs, ignore_index=True)
    output_path = output_dir / "tracking.parquet"
    tracking_combined.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"✓ tracking.parquet created: {len(tracking_combined)} rows")
    return output_path


def combine_all(
    data_dir: Path = Path("data"), output_dir: Path | None = None
) -> dict[str, Path]:
    output_dir = output_dir or data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "events": combine_events(data_dir, output_dir),
        "shifts": combine_shifts(data_dir, output_dir),
        "tracking": combine_tracking(data_dir, output_dir),
    }
    print("\nDone!")
    return outputs


if __name__ == "__main__":
    combine_all()
