"""Patient and joint splits for PyHealth samples."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import dask.dataframe as dd
import pandas as pd

from .dataset import (
    TABLE_ROOT,
    get_default_tables,
    load_subject_splits,
    make_fixed_split_datasets,
)


VAL_TIMESTAMP = pd.Timestamp("2123-01-01")
TEST_TIMESTAMP = pd.Timestamp("2125-01-01")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_PROTOCOLS = ("patient", "joint")
LABEL_OBSERVATION_END_KEY = "label_observation_end"
ADMISSION_LABEL_TASKS = {
    "digestive_subtype",
    "cardiovascular_subtype",
    "phenotyping",
}

SOURCE_TIME_COLUMNS = {
    "admissions": "admittime",
    "admission_discharges": "dischtime",
    "diagnoses_icd": "diagnosis_time",
    "procedures_icd": "procedure_time",
    "prescriptions": "starttime",
    "prescription_stops": "stoptime",
    "labevents": "lab_time",
    "drgcodes": "drg_time",
    "omr": "omr_time",
}


def get_tables_for_split_protocol(
    task_name: str,
    split_protocol: str,
) -> list[str]:
    """Return base task tables plus joint-only completion data when needed."""
    if split_protocol not in SPLIT_PROTOCOLS:
        raise ValueError(
            f"Unknown split_protocol={split_protocol!r}. "
            f"Available protocols: {list(SPLIT_PROTOCOLS)}"
        )

    tables = get_default_tables(task_name)
    if (
        split_protocol == "joint"
        and task_name in ADMISSION_LABEL_TASKS
        and "admission_discharges" not in tables
    ):
        tables.append("admission_discharges")
    return tables


@dataclass(frozen=True)
class JointSplitWindow:
    """Calendar and observation boundaries for one joint split."""

    evaluation_start: pd.Timestamp | None
    observation_end: pd.Timestamp
    end_inclusive: bool


def get_joint_split_window(
    split_name: str,
    database_end: pd.Timestamp,
) -> JointSplitWindow:
    """Return the RelBench-compatible time window for one joint split."""
    database_end = pd.Timestamp(database_end)
    if pd.isna(database_end):
        raise ValueError("database_end must be a valid timestamp")

    if split_name == "train":
        return JointSplitWindow(None, VAL_TIMESTAMP, False)
    if split_name == "val":
        return JointSplitWindow(VAL_TIMESTAMP, TEST_TIMESTAMP, False)
    if split_name == "test":
        return JointSplitWindow(TEST_TIMESTAMP, database_end, True)
    raise ValueError(f"Unknown split_name={split_name!r}")


@lru_cache(maxsize=None)
def get_source_database_end(root: str | Path = TABLE_ROOT) -> pd.Timestamp:
    """Return the maximum timestamp across all time-bearing source tables."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MIMIC table directory not found at {root}")

    observed_ends: list[pd.Timestamp] = []
    for table_name, time_column in SOURCE_TIME_COLUMNS.items():
        table_path = root / f"{table_name}.parquet"
        if not table_path.is_file():
            continue

        frame = dd.read_parquet(table_path, columns=[time_column])
        table_end = dd.to_datetime(frame[time_column], errors="coerce").max().compute()
        if pd.notna(table_end):
            observed_ends.append(pd.Timestamp(table_end))

    if not observed_ends:
        raise RuntimeError(f"No observable timestamps found under {root}")
    return max(observed_ends)


def sample_belongs_to_joint_window(
    sample: Mapping[str, Any],
    split_name: str,
    database_end: pd.Timestamp,
) -> bool:
    """Check evaluation time and complete label observation for a joint split."""
    if "timestamp" not in sample:
        raise KeyError("Joint split samples must contain 'timestamp'.")
    if LABEL_OBSERVATION_END_KEY not in sample:
        raise KeyError(
            "Joint split samples must contain "
            f"{LABEL_OBSERVATION_END_KEY!r}; rebuild the joint task cache."
        )

    evaluation_time = pd.to_datetime(sample["timestamp"], errors="coerce")
    label_end = pd.to_datetime(sample[LABEL_OBSERVATION_END_KEY], errors="coerce")
    if pd.isna(evaluation_time) or pd.isna(label_end):
        return False

    window = get_joint_split_window(split_name, database_end)
    if (
        window.evaluation_start is not None
        and evaluation_time < window.evaluation_start
    ):
        return False

    if window.end_inclusive:
        return (
            evaluation_time <= window.observation_end
            and label_end <= window.observation_end
        )
    return (
        evaluation_time < window.observation_end and label_end < window.observation_end
    )


def subset_by_joint_split(
    sample_dataset,
    patient_ids: Iterable[int],
    split_name: str,
    database_end: pd.Timestamp,
):
    """Create a sample subset satisfying both patient and temporal rules."""
    indices: list[int] = []

    for patient_id in patient_ids:
        patient_indices = sample_dataset.patient_to_index.get(str(patient_id), [])
        joint_indices = [
            index
            for index in patient_indices
            if sample_belongs_to_joint_window(
                sample_dataset[index], split_name, database_end
            )
        ]
        indices.extend(joint_indices)

    indices.sort()
    if not indices:
        raise ValueError(
            f"No samples found for joint split {split_name}. "
            "Check patient IDs, the joint task cache, and time cutoffs."
        )
    return sample_dataset.subset(indices)


def make_joint_split_datasets(
    sample_dataset,
    split_root: str | Path,
    table_root: str | Path = TABLE_ROOT,
):
    """Build RelBench-style joint patient-and-time sample splits."""
    subject_splits = load_subject_splits(split_root)
    database_end = get_source_database_end(table_root)
    return {
        split_name: subset_by_joint_split(
            sample_dataset,
            subject_splits[split_name],
            split_name,
            database_end,
        )
        for split_name in SPLIT_NAMES
    }


def make_split_datasets(
    sample_dataset,
    split_protocol: str = "patient",
    split_root: str | Path | None = None,
    table_root: str | Path = TABLE_ROOT,
):
    """Dispatch to the existing patient split or the isolated joint split."""
    if split_root is None:
        raise ValueError("split_root is required")
    if split_protocol == "patient":
        return make_fixed_split_datasets(sample_dataset, split_root=split_root)
    if split_protocol == "joint":
        return make_joint_split_datasets(
            sample_dataset,
            split_root=split_root,
            table_root=table_root,
        )
    raise ValueError(
        f"Unknown split_protocol={split_protocol!r}. "
        f"Available protocols: {list(SPLIT_PROTOCOLS)}"
    )
