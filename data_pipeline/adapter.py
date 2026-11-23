from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io import (
    PIPELINE_VERSION,
    ensure_empty_or_matching,
    load_table,
    read_json,
    stable_hash,
    utc_now,
    write_json,
)
from .paths import DEFAULT_DATA_ROOT
from .schema import DEFAULT_RELBENCH_TABLES, RELBENCH_TABLES


@dataclass(frozen=True)
class RelBenchAdaptConfig:
    name: str = "relehr-hosp-v1"
    subset_dir: Path = DEFAULT_DATA_ROOT / "subsets" / "hosp_subset_v1"
    tables: tuple[str, ...] = DEFAULT_RELBENCH_TABLES
    data_root: Path = DEFAULT_DATA_ROOT

    @property
    def out_dir(self) -> Path:
        return Path(self.data_root) / "relbench" / self.name

    def manifest_payload(self) -> dict[str, Any]:
        subset_manifest = read_json(Path(self.subset_dir) / "manifest.json")
        return {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "relbench_adapter",
            "source_subset_dir": str(Path(self.subset_dir).resolve()),
            "source_subset_config_hash": subset_manifest.get("config_hash"),
            "config": asdict(self),
        }


class RelBenchAdapter:
    def __init__(self, config: RelBenchAdaptConfig):
        self.config = config

    def run(self, force: bool = False) -> Path:
        manifest = self.config.manifest_payload()
        manifest["config_hash"] = stable_hash(
            manifest["config"]
            | {
                "pipeline_version": PIPELINE_VERSION,
                "source_subset_config_hash": manifest["source_subset_config_hash"],
            }
        )
        manifest["created_at"] = utc_now()
        if ensure_empty_or_matching(self.config.out_dir, manifest, force):
            print(f"RelBench cache already exists: {self.config.out_dir}")
            return self.config.out_dir

        db = self._make_database()
        mappings = self._reindex_with_mappings(db)
        db_path = self.config.out_dir / "db"
        db_path.mkdir(parents=True, exist_ok=True)
        db.save(db_path)

        mappings_dir = self.config.out_dir / "mappings"
        mappings_dir.mkdir(parents=True, exist_ok=True)
        mapping_rows = {}
        for table_name, mapping in mappings.items():
            mapping_rows[table_name] = len(mapping)
            mapping.to_parquet(mappings_dir / f"{table_name}.parquet", index=False)

        manifest["rows"] = {
            name: len(table.df) for name, table in db.table_dict.items()
        }
        manifest["mapping_rows"] = mapping_rows
        manifest["table_schema"] = {
            name: {
                "pkey_col": spec.pkey_col,
                "time_col": spec.time_col,
                "fkey_col_to_pkey_table": spec.fkey_col_to_pkey_table,
            }
            for name, spec in RELBENCH_TABLES.items()
            if name in db.table_dict
        }
        write_json(self.config.out_dir / "manifest.json", manifest)
        write_json(self.config.out_dir / "table_schema.json", manifest["table_schema"])
        return self.config.out_dir

    def _make_database(self):
        try:
            from relbench.base import Database, Table
        except ImportError as exc:
            raise RuntimeError(
                "Could not import relbench. Install the data/RelBench requirements "
                "and activate that environment."
            ) from exc

        table_dict = {}
        table_dir = Path(self.config.subset_dir) / "tables"
        for table_name in self.config.tables:
            path = table_dir / f"{table_name}.parquet"
            if not path.exists():
                continue
            spec = RELBENCH_TABLES[table_name]
            df = load_table(path)
            table_dict[table_name] = Table(
                df=df,
                fkey_col_to_pkey_table=dict(spec.fkey_col_to_pkey_table),
                pkey_col=spec.pkey_col,
                time_col=spec.time_col,
            )
        missing_required = {"patients", "admissions"} - set(table_dict)
        if missing_required:
            raise RuntimeError(
                f"Subset is missing required tables: {sorted(missing_required)}"
            )
        missing_fkeys = []
        for table_name, table in table_dict.items():
            for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
                if pkey_table not in table_dict:
                    missing_fkeys.append(f"{table_name}.{fkey_col} -> {pkey_table}")
        if missing_fkeys:
            raise RuntimeError(
                "Requested RelBench tables have missing foreign-key target tables: "
                + ", ".join(sorted(missing_fkeys))
            )
        return Database(table_dict)

    def _reindex_with_mappings(self, db) -> dict[str, pd.DataFrame]:
        original_cols: dict[str, str] = {}
        for table_name, table in db.table_dict.items():
            if table.pkey_col is None:
                continue
            orig_col = f"__orig_{table.pkey_col}"
            original_cols[table_name] = orig_col
            table.df = table.df.copy()
            table.df[orig_col] = table.df[table.pkey_col]

        db.reindex_pkeys_and_fkeys()

        mappings: dict[str, pd.DataFrame] = {}
        for table_name, orig_col in original_cols.items():
            table = db.table_dict[table_name]
            pkey_col = table.pkey_col
            mapping = table.df[[pkey_col, orig_col]].copy()
            mapping = mapping.rename(
                columns={
                    pkey_col: "relbench_id",
                    orig_col: f"original_{pkey_col}",
                }
            )
            mappings[table_name] = mapping
            table.df = table.df.drop(columns=[orig_col])
        return mappings
