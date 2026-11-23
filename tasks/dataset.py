"""Cache-only RelBench datasets with patient and joint splits."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from relbench.base import Database, Dataset, Table


PATIENT_DATASET_NAME = "relehrbench-mimic-patient"
JOINT_DATASET_NAME = "relehrbench-mimic-joint"
VAL_TIMESTAMP = pd.Timestamp("2123-01-01")
TEST_TIMESTAMP = pd.Timestamp("2125-01-01")
SPLIT_NAMES = ("train", "val", "test")


def _require_cached_db(cache_dir: str | None) -> Path:
    if cache_dir is None:
        raise FileNotFoundError(
            "A RelBench cache directory containing db/ is required."
        )
    db_path = Path(cache_dir) / "db"
    if not db_path.is_dir() or not any(db_path.glob("*.parquet")):
        raise FileNotFoundError(
            f"RelBench database not found at {db_path}. Run the data pipeline first."
        )
    return db_path


class MimicPatientSplitDataset(Dataset):
    """MIMIC-IV cache with a deterministic patient-level split."""

    # EntityTask requires temporal cutoffs even though every submitted task
    # overrides _get_table() and uses the patient split directly.
    val_timestamp = VAL_TIMESTAMP
    test_timestamp = TEST_TIMESTAMP

    def __init__(
        self,
        cache_dir: str | None = None,
        split_id: str = "v2_2_80_10_10_seed42_numpy",
        split_seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        _require_cached_db(cache_dir)
        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(
                "split_ratios must contain train/val/test ratios summing to 1"
            )
        self.split_id = split_id
        self.split_seed = split_seed
        self.split_ratios = split_ratios
        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        _require_cached_db(self.cache_dir)
        raise RuntimeError("The validated database cache should have been loaded.")

    @property
    def patient_split_root(self) -> Path:
        return Path(self.cache_dir) / "db_patient" / self.split_id

    @property
    def original_split_dir(self) -> Path:
        return self.patient_split_root / "original_subject_ids"

    def get_patient_split(self) -> dict[str, np.ndarray]:
        """Generate once, then load a deterministic 80/10/10-style split."""

        split_dir = self.patient_split_root / "subject_ids"
        split_paths = {name: split_dir / f"{name}.npy" for name in SPLIT_NAMES}
        if all(path.exists() for path in split_paths.values()):
            splits = {name: np.load(path) for name, path in split_paths.items()}
            self._save_original_subject_ids(splits)
            return splits

        db = self.get_db(upto_test_timestamp=False)
        patient_ids = (
            db.table_dict["patients"].df["subject_id"].dropna().to_numpy(dtype=np.int64)
        )
        discharges = db.table_dict["admission_discharges"].df
        eligible_ids = (
            discharges.loc[discharges["dischtime"].notna(), "subject_id"]
            .dropna()
            .unique()
            .astype(np.int64)
        )
        patient_ids = np.intersect1d(patient_ids, eligible_ids)
        permutation = np.random.default_rng(self.split_seed).permutation(patient_ids)

        num_train = int(self.split_ratios[0] * len(permutation))
        num_val = int(self.split_ratios[1] * len(permutation))
        splits = {
            "train": np.sort(permutation[:num_train]),
            "val": np.sort(permutation[num_train : num_train + num_val]),
            "test": np.sort(permutation[num_train + num_val :]),
        }

        split_dir.mkdir(parents=True, exist_ok=True)
        for name, ids in splits.items():
            np.save(split_paths[name], ids)
        self._save_original_subject_ids(splits)
        return splits

    def _save_original_subject_ids(self, splits: dict[str, np.ndarray]) -> None:
        output_paths = {
            name: self.original_split_dir / f"{name}.npy" for name in SPLIT_NAMES
        }
        if all(path.exists() for path in output_paths.values()):
            return

        mapping = pd.read_parquet(
            Path(self.cache_dir) / "mappings" / "patients.parquet"
        ).set_index("relbench_id")["original_subject_id"]
        self.original_split_dir.mkdir(parents=True, exist_ok=True)
        for name, relbench_ids in splits.items():
            original_ids = mapping.loc[relbench_ids].to_numpy(dtype=np.int64)
            np.save(output_paths[name], original_ids)

    @staticmethod
    def _filter_tables_by_subjects(db: Database, keep: Iterable[int]) -> Database:
        keep_ids = set(int(value) for value in keep)
        tables = {}
        for name, table in db.table_dict.items():
            df = table.df
            if "subject_id" in df.columns:
                df = df[df["subject_id"].isin(keep_ids)]
            tables[name] = Table(
                df=df.reset_index(drop=True).copy(),
                fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
        return Database(tables)

    @staticmethod
    def _reindex(db: Database) -> None:
        db.reindex_pkeys_and_fkeys()

    def get_split_db(
        self,
        split: str,
        upto_test_timestamp: bool = False,
    ) -> Database:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split={split!r}")

        cache = self.patient_split_root / split
        success_marker = cache / "_SUCCESS"
        if success_marker.exists() and any(cache.glob("*.parquet")):
            return Database.load(cache)

        full_db = self.get_db(upto_test_timestamp=upto_test_timestamp)
        split_db = self._filter_tables_by_subjects(
            full_db, self.get_patient_split()[split]
        )
        self._reindex(split_db)
        cache.mkdir(parents=True, exist_ok=True)
        split_db.save(cache)
        success_marker.touch()
        return split_db


class MimicJointSplitDataset(MimicPatientSplitDataset):
    """The same patient cohorts with cumulative temporal history windows."""

    is_joint_split = True

    @property
    def end_timestamp(self) -> pd.Timestamp:
        full_db = self.get_db(upto_test_timestamp=False)
        observed_ends = [
            pd.to_datetime(table.df[table.time_col], errors="coerce").max()
            for table in full_db.table_dict.values()
            if table.time_col is not None
        ]
        observed_ends = [value for value in observed_ends if pd.notna(value)]
        if not observed_ends:
            raise RuntimeError("The database has no observable timestamps.")
        return max(observed_ends)

    def _filter_tables_by_time(self, db: Database, split: str) -> Database:
        tables = {}
        for name, table in db.table_dict.items():
            df = table.df
            if table.time_col is not None:
                timestamps = pd.to_datetime(df[table.time_col], errors="coerce")
                if split == "train":
                    df = df.loc[timestamps < self.val_timestamp]
                elif split == "val":
                    df = df.loc[timestamps < self.test_timestamp]
                else:
                    df = df.loc[timestamps <= self.end_timestamp]
            tables[name] = Table(
                df=df.reset_index(drop=True).copy(),
                fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
        return Database(tables)

    def get_split_db(
        self,
        split: str,
        upto_test_timestamp: bool = False,
    ) -> Database:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split={split!r}")
        del upto_test_timestamp

        cache = Path(self.cache_dir) / "db_joint" / self.split_id / split
        success_marker = cache / "_SUCCESS"
        if success_marker.exists() and any(cache.glob("*.parquet")):
            return Database.load(cache)

        full_db = self.get_db(upto_test_timestamp=False)
        split_db = self._filter_tables_by_subjects(
            full_db, self.get_patient_split()[split]
        )
        split_db = self._filter_tables_by_time(split_db, split)
        self._reindex(split_db)
        cache.mkdir(parents=True, exist_ok=True)
        split_db.save(cache)
        success_marker.touch()
        return split_db


# class MimicTemporalDataset(Dataset):
#     """MIMIC-IV v2.2 cache with global temporal validation/test cutoffs."""
#     val_timestamp = VAL_TIMESTAMP
#     test_timestamp = TEST_TIMESTAMP

#     def __init__(self, cache_dir: str | None = None) -> None:
#         _require_cached_db(cache_dir)
#         super().__init__(cache_dir=cache_dir)

#     def make_db(self) -> Database:
#         _require_cached_db(self.cache_dir)
#         raise RuntimeError("Unreachable: the validated MIMIC v2.2 cache must be loaded.")
