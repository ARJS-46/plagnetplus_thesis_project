from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "crystal_id", "locality", "sample_id",
    "scan_id", "relative_path", "split",
}

def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Manifest must be CSV or Parquet.")

def validate_manifest(frame: pd.DataFrame) -> list[str]:
    errors = []
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    if "crystal_id" in frame and frame["crystal_id"].duplicated().any():
        errors.append("crystal_id values are not unique.")
    if "relative_path" in frame and frame["relative_path"].duplicated().any():
        errors.append("relative_path values are not unique.")
    return errors
