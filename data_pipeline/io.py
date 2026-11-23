import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_VERSION = "mimic4-v2"

_END_OF_DAY = pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_datetime(values: pd.Series) -> pd.Series:
    """Parse timestamps, placing date-only values at the end of their day."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    date_only = values.map(_has_date_only_resolution)
    return parsed.mask(date_only, parsed + _END_OF_DAY)


def _has_date_only_resolution(value: Any) -> bool:
    if isinstance(value, datetime):
        return False
    if isinstance(value, date):
        return True
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value.strip())
    except ValueError:
        return False
    return True


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    return value


def stable_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_empty_or_matching(path: Path, manifest: dict[str, Any], force: bool) -> bool:
    """Skip because an identical cache exists."""
    manifest_path = path / "manifest.json"
    if path.exists() and force:
        shutil.rmtree(path)
        return False
    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        if existing.get("config_hash") == manifest.get("config_hash"):
            return True
        raise RuntimeError(
            f"Cache exists at {path} with config_hash={existing.get('config_hash')}, "
            f"but requested config_hash={manifest.get('config_hash')}. "
            "Use --force or choose another --name."
        )
    if path.exists() and any(path.iterdir()) and not force:
        raise RuntimeError(
            f"Cache directory {path} is not empty and has no matching manifest. "
            "Use --force or choose another --name."
        )
    return False


def convert_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        lower = col.lower()
        if "time" in lower or "date" in lower:
            try:
                df[col] = parse_datetime(df[col])
            except Exception:
                pass
        elif str(df[col].dtype) == "object":
            non_null = df[col].dropna()
            if len(non_null) == 0 or all(isinstance(v, str) for v in non_null):
                df[col] = df[col].astype("string")
    return df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    convert_for_parquet(df).to_parquet(path, index=False)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
