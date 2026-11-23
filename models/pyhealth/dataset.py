from __future__ import annotations

from pathlib import Path
from typing import Iterable

import dask.dataframe as dd
import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset  # type: ignore[reportMissingImports]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET_CACHE_NAME = "hosp_subset_v2_2_30k"
RELBENCH_CACHE_NAME = "relehr-v2_2_30k"
TABLE_ROOT = (
    PROJECT_ROOT
    / "cache"
    / "datasets"
    / "subsets"
    / DEFAULT_SUBSET_CACHE_NAME
    / "tables"
)
CONFIG_PATH = Path(__file__).with_name("config.yaml")

READMISSION_TABLES = [
    "patients",
    "admissions",
    "admission_discharges",
    "diagnoses_icd",
    "procedures_icd",
    "prescriptions",
    "labevents",
]

DIGESTIVE_SUBTYPE_TABLES = [
    "patients",
    "admissions",
    "diagnoses_icd",
    "procedures_icd",
    "prescriptions",
    "labevents",
]

ADMISSION_MULTILABEL_TABLES = DIGESTIVE_SUBTYPE_TABLES

TASK_TABLES = {
    "readmission_30d": READMISSION_TABLES,
    "high_risk_lab_24h": READMISSION_TABLES,
    "digestive_subtype": DIGESTIVE_SUBTYPE_TABLES,
    "cardiovascular_subtype": ADMISSION_MULTILABEL_TABLES,
    "phenotyping": ADMISSION_MULTILABEL_TABLES,
}


class MimicHospPyHealthDataset(BaseDataset):
    """PyHealth dataset adaptor for local MIMIC-Hosp parquet tables."""

    dataset_name = "mimic-hosp-30k-pyhealth"

    def __init__(
        self,
        cache_dir: str | Path,
        root: str | Path = TABLE_ROOT,
        tables: Iterable[str] | None = None,
        config_path: str | Path = CONFIG_PATH,
        dev: bool = True,
        num_workers: int = 1,
    ) -> None:
        if tables is None:
            tables = READMISSION_TABLES

        super().__init__(
            root=str(root),
            tables=list(tables),
            config_path=str(config_path),
            cache_dir=str(cache_dir),
            dev=dev,
            num_workers=num_workers,
        )

    def _scan_csv_tsv_gz(self, source_path: str) -> dd.DataFrame:
        """Support parquet in addition to PyHealth's CSV/TSV loader."""
        source_path = str(Path(source_path).expanduser().resolve())

        if source_path.lower().endswith(".parquet"):
            df = dd.read_parquet(
                source_path,
                split_row_groups=True,
                blocksize="64MB",
            )
            return df.replace("", pd.NA)

        return super()._scan_csv_tsv_gz(source_path)


def get_default_tables(task_name: str | None = None) -> list[str]:
    """Return the default table list for a supported task."""
    if task_name is None:
        return list(READMISSION_TABLES)
    if task_name not in TASK_TABLES:
        raise KeyError(
            f"Unknown task_name={task_name!r}. Available tasks: {sorted(TASK_TABLES)}"
        )
    return list(TASK_TABLES[task_name])


def _normalize_lab_label(label: object) -> str:
    return " ".join(str(label).strip().lower().split())


def resolve_lab_itemids(
    labels: Iterable[str],
    root: str | Path = TABLE_ROOT,
) -> frozenset[int]:
    """Resolve lab names through d_labitems without exposing it as a feature."""
    path = Path(root) / "d_labitems.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Lab dictionary not found at {path}; it is required for label construction."
        )

    d_labitems = pd.read_parquet(path, columns=["itemid", "label"])
    target_labels = {_normalize_lab_label(label) for label in labels}
    mask = d_labitems["label"].map(_normalize_lab_label).isin(target_labels)
    itemids = pd.to_numeric(d_labitems.loc[mask, "itemid"], errors="coerce").dropna()
    resolved = frozenset(itemids.astype(int).tolist())
    if not resolved:
        raise RuntimeError(
            "None of the configured high-risk lab labels were found in d_labitems."
        )
    return resolved


def get_split_root(split_id: str) -> Path:
    return (
        PROJECT_ROOT
        / "cache"
        / "datasets"
        / "relbench"
        / RELBENCH_CACHE_NAME
        / "db_patient"
        / split_id
        / "original_subject_ids"
    )


def load_subject_splits(split_root: str | Path) -> dict[str, set[int]]:
    """Load fixed train/val/test subject-id splits."""
    split_root = Path(split_root)
    return {
        name: set(np.load(split_root / f"{name}.npy").astype(int).tolist())
        for name in ["train", "val", "test"]
    }


def subset_by_patient_ids(sample_dataset, patient_ids, split_name: str):
    """Create a PyHealth sample subset from fixed subject IDs."""
    indices = []

    for patient_id in patient_ids:
        patient_key = str(patient_id)
        patient_indices = sample_dataset.patient_to_index.get(patient_key, [])
        indices.extend(patient_indices)

    indices = sorted(indices)

    if not indices:
        raise ValueError(
            f"No samples found for split {split_name}. "
            "Check dev=True cohort truncation and patient id normalization."
        )

    return sample_dataset.subset(indices)


def make_fixed_split_datasets(sample_dataset, split_root: str | Path):
    """Split a PyHealth SampleDataset using the fixed subject-level split."""
    split_root = Path(split_root)
    train_ids = np.load(split_root / "train.npy")
    val_ids = np.load(split_root / "val.npy")
    test_ids = np.load(split_root / "test.npy")

    return {
        "train": subset_by_patient_ids(sample_dataset, train_ids, "train"),
        "val": subset_by_patient_ids(sample_dataset, val_ids, "val"),
        "test": subset_by_patient_ids(sample_dataset, test_ids, "test"),
    }
