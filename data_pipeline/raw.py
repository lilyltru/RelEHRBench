import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io import (
    PIPELINE_VERSION,
    ensure_empty_or_matching,
    save_table,
    stable_hash,
    utc_now,
    write_json,
)
from .paths import DEFAULT_DATA_ROOT
from .schema import (
    DEFAULT_RAW_TABLES,
    MIMIC_VERSION,
    RAW_TABLES,
    RawTableSpec,
)


@dataclass(frozen=True)
class RawExtractConfig:
    name: str = "mimic4_raw_v1"
    tables: tuple[str, ...] = DEFAULT_RAW_TABLES
    dataset_name: str = "physionet-data"
    location: str = "US"
    data_root: Path = DEFAULT_DATA_ROOT

    @property
    def out_dir(self) -> Path:
        return Path(self.data_root) / "raw" / self.name

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "raw",
            "mimic_version": MIMIC_VERSION,
            "config": asdict(self),
        }


class MimicRawExtractor:
    def __init__(self, config: RawExtractConfig, project_id: str | None = None):
        from google.cloud import bigquery

        self.config = config
        self.project_id = project_id or os.getenv("PROJECT_ID")
        if not self.project_id:
            raise ValueError("Set PROJECT_ID or pass project_id explicitly.")
        self.bigquery = bigquery
        self.client = bigquery.Client(project=self.project_id, location=config.location)

    def run(self, force: bool = False) -> Path:
        requested = self._requested_specs()

        manifest = self.config.manifest_payload()
        manifest["config_hash"] = stable_hash(
            manifest["config"] | {"pipeline_version": PIPELINE_VERSION}
        )
        manifest["created_at"] = utc_now()
        if ensure_empty_or_matching(self.config.out_dir, manifest, force):
            print(f"Raw cache already exists: {self.config.out_dir}")
            return self.config.out_dir

        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        extracted: dict[str, pd.DataFrame] = {}
        for table_name, spec in requested.items():
            df = self._query_full_table(spec)
            extracted[table_name] = df
            self._save_raw_table(spec, df)

        if "patients" in extracted and "subject_id" in extracted["patients"].columns:
            subject_ids = _format_ids(extracted["patients"]["subject_id"])
            save_table(
                pd.DataFrame({"subject_id": subject_ids}),
                self.config.out_dir / "cohort" / "subject_ids.parquet",
            )

        if "admissions" in extracted and "hadm_id" in extracted["admissions"].columns:
            hadm_ids = _format_ids(extracted["admissions"]["hadm_id"])
            save_table(
                pd.DataFrame({"hadm_id": hadm_ids}),
                self.config.out_dir / "cohort" / "hadm_ids.parquet",
            )
        manifest["rows"] = self._row_counts()
        write_json(self.config.out_dir / "manifest.json", manifest)
        return self.config.out_dir

    def _requested_specs(self) -> dict[str, RawTableSpec]:
        unsupported_tables = set(self.config.tables) - set(RAW_TABLES)
        if unsupported_tables:
            raise ValueError(f"Unsupported raw tables: {sorted(unsupported_tables)}")

        requested = {name: RAW_TABLES[name] for name in self.config.tables}
        if not requested:
            raise ValueError("No raw tables selected for the requested modules.")
        return requested

    def _run_query(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        job_config = self.bigquery.QueryJobConfig(query_parameters=params or [])
        return self.client.query_and_wait(query, job_config=job_config).to_dataframe()

    def _query_full_table(self, spec: RawTableSpec) -> pd.DataFrame:
        query = f"""
        SELECT *
        FROM `{self.config.dataset_name}.{spec.dataset}.{spec.table}`
        ORDER BY subject_id
        """
        return _sanitize_bq_extension_dtypes(self._run_query(query))

    def _save_raw_table(self, spec: RawTableSpec, df: pd.DataFrame) -> None:
        save_table(df, self.config.out_dir / spec.module / f"{spec.table}.parquet")

    def _row_counts(self) -> dict[str, dict[str, int]]:
        counts = {}
        for module_dir in sorted(self.config.out_dir.glob("*")):
            if not module_dir.is_dir() or module_dir.name == "cohort":
                continue
            counts[module_dir.name] = {
                table_path.stem: len(pd.read_parquet(table_path))
                for table_path in sorted(module_dir.glob("*.parquet"))
            }
        return counts


def _format_ids(series: pd.Series) -> list[int]:
    return [int(value) for value in series.dropna().unique().tolist()]


def _sanitize_bq_extension_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BigQuery-specific DATE/TIME extension dtypes to plain objects,
    so that the cached Parquet files do not depend on db_dtypes.
    """
    for col in df.columns:
        if str(df[col].dtype) in ("dbdate", "dbtime"):
            df[col] = df[col].astype(object)
    return df
