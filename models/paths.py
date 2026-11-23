from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


def model_cache_dir(
    model: str,
    protocol: str,
    split_id: str,
    task: str,
) -> Path:
    return SUBMISSION_ROOT / "cache" / model / protocol / split_id / task


def new_run_dir(
    model: str,
    protocol: str,
    split_id: str,
    task: str,
    seed: int,
) -> tuple[Path, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    run_dir = (
        SUBMISSION_ROOT
        / "outputs"
        / model
        / protocol
        / split_id
        / task
        / f"seed_{seed}"
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id
