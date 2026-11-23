import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io import (
    PIPELINE_VERSION,
    ensure_empty_or_matching,
    load_table,
    parse_datetime,
    read_json,
    save_table,
    stable_hash,
    utc_now,
    write_json,
)
from .paths import DEFAULT_DATA_ROOT
from .schema import DEFAULT_RELBENCH_TABLES, MIMIC_VERSION


@dataclass(frozen=True)
class HospSubsetConfig:
    name: str = "hosp_subset_v1"
    raw_dir: Path = DEFAULT_DATA_ROOT / "raw" / "mimic4_raw_v1"
    tables: tuple[str, ...] = DEFAULT_RELBENCH_TABLES
    patients_limit: int = 30000
    min_age: int | None = 18
    require_admittime: bool = True
    require_dischtime: bool = True
    cohort_source: str = "admissions"
    require_prescription_starttime: bool = True
    require_prescription_stoptime: bool = True
    require_admittime_lessthan_discharge_time: bool = True
    require_numeric_labs: bool = True
    top_lab_items: int = 100
    dataset_name: str = "physionet-data"
    location: str = "US"
    data_root: Path = DEFAULT_DATA_ROOT

    @property
    def out_dir(self) -> Path:
        return Path(self.data_root) / "subsets" / self.name

    def manifest_payload(self) -> dict[str, Any]:
        raw_manifest = read_json(Path(self.raw_dir) / "manifest.json")
        return {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "subset",
            "source_raw_dir": str(Path(self.raw_dir).resolve()),
            "source_raw_config_hash": raw_manifest.get("config_hash"),
            "config": asdict(self),
        }


class HospSubsetBuilder:
    def __init__(self, config: HospSubsetConfig, project_id: str | None = None):
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.bigquery = None
        self.client = None

    def run(self, force: bool = False) -> Path:
        manifest = self.config.manifest_payload()
        manifest["config_hash"] = stable_hash(
            manifest["config"]
            | {
                "pipeline_version": PIPELINE_VERSION,
                "source_raw_config_hash": manifest["source_raw_config_hash"],
            }
        )
        manifest["created_at"] = utc_now()
        if ensure_empty_or_matching(self.config.out_dir, manifest, force):
            print(f"Subset cache already exists: {self.config.out_dir}")
            return self.config.out_dir

        raw = self._load_raw_tables()
        subset = self._build_base_tables(raw) # patient + admission table
        query_manifest = self._add_hosp_query_tables(subset) # more clinical tables

        for table_name, df in subset.items():
            if table_name in self.config.tables:
                save_table(df, self.config.out_dir / "tables" / f"{table_name}.parquet")

        self._save_cohort_ids(subset)
        manifest.update(query_manifest)
        manifest["rows"] = {
            name: len(df) for name, df in subset.items() if name in self.config.tables
        }
        manifest["cohort"] = self._cohort_summary(subset)
        write_json(self.config.out_dir / "manifest.json", manifest)
        return self.config.out_dir

    def _load_raw_tables(self) -> dict[str, pd.DataFrame]:
        tables = {}
        for path in sorted((self.raw_dir / "hosp").glob("*.parquet")):
            tables[path.stem] = load_table(path)
        missing = {"patients", "admissions"} - set(tables)
        if missing:
            raise RuntimeError(
                f"Raw cache is missing required tables: {sorted(missing)}"
            )
        return tables

    def _build_base_tables(
        self, raw: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        # extract cohort satisfying constraints: age, admittime, discharge time
        patients = raw["patients"].copy()
        admissions_raw = raw["admissions"].copy()

        if self.config.min_age is not None and "anchor_age" in patients.columns:
            patients = patients[
                pd.to_numeric(patients["anchor_age"], errors="coerce")
                >= self.config.min_age
            ]

        for time_col in ("admittime", "dischtime"):
            if time_col in admissions_raw.columns:
                admissions_raw[time_col] = parse_datetime(admissions_raw[time_col])

        if self.config.require_admittime and "admittime" in admissions_raw.columns:
            admissions_raw = admissions_raw[admissions_raw["admittime"].notna()]

        if self.config.require_admittime_lessthan_discharge_time and {
            "admittime",
            "dischtime",
        }.issubset(admissions_raw.columns):
            invalid_window = (
                admissions_raw["admittime"].notna()
                & admissions_raw["dischtime"].notna()
                & (admissions_raw["dischtime"] < admissions_raw["admittime"])
            )
            admissions_raw = admissions_raw[~invalid_window]

        patients = self._select_patient_cohort(patients, admissions_raw)
        admissions_raw = admissions_raw[
            admissions_raw["subject_id"].isin(patients["subject_id"])
        ]
        discharges_raw = admissions_raw
        if self.config.require_dischtime and "dischtime" in discharges_raw.columns:
            discharges_raw = discharges_raw[discharges_raw["dischtime"].notna()]

        tables: dict[str, pd.DataFrame] = {}
        # tables["patients"] = _sort_existing(
        #     _select_existing(patients, ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]),
        #     ["subject_id"],
        # )
        tables["patients"] = _sort_existing(
            _select_existing(patients, ["subject_id", "gender", "anchor_age"]),
            ["subject_id"],
        )

        tables["admission_discharges"] = _sort_existing(
            _select_existing(
                discharges_raw,
                ["subject_id", "hadm_id", "dischtime", "discharge_location"],
            ),
            ["subject_id", "hadm_id"],
        )

        tables["admissions"] = _sort_existing(
            _select_existing(
                admissions_raw,
                [
                    "subject_id",
                    "hadm_id",
                    "admittime",
                    "admission_type",
                    "admission_location",
                    "insurance",
                    "language",
                    "marital_status",
                    "race",
                ],
            ),
            ["subject_id", "admittime", "hadm_id"],
        )
        return tables

    def _select_patient_cohort(
        self, patients: pd.DataFrame, admissions: pd.DataFrame
    ) -> pd.DataFrame:
        if self.config.cohort_source == "patients":
            cohort = patients.sort_values("subject_id")
        elif self.config.cohort_source == "admissions":
            # satisfy age requirements + at least one no-null admittime
            if "admittime" not in admissions.columns:
                raise RuntimeError(
                    "admissions.admittime is required for admissions cohort_source."
                )
            first_admission = (
                admissions.groupby("subject_id", as_index=False)["admittime"]
                .min()
                .rename(columns={"admittime": "first_admittime"})
            )
            cohort = patients.merge(first_admission, on="subject_id", how="inner")
            cohort = cohort.sort_values(["first_admittime", "subject_id"])
            cohort = cohort.drop(columns=["first_admittime"])
        else:
            raise ValueError("cohort_source must be 'patients' or 'admissions'.")

        if self.config.patients_limit > 0:
            cohort = cohort.head(self.config.patients_limit)
        return cohort.reset_index(drop=True)

    def _add_hosp_query_tables(self, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
        requested = set(self.config.tables)
        subject_ids = _format_ids(tables["patients"]["subject_id"])
        hadm_ids = _format_ids(tables["admissions"]["hadm_id"])
        query_manifest: dict[str, Any] = {}

        lab_itemids: list[int] = []
        if requested & {"labevents", "d_labitems"}:
            lab_itemids = self._query_top_lab_itemids(subject_ids, hadm_ids)
            query_manifest["top_lab_itemids"] = lab_itemids
        if "labevents" in requested:
            raw_labs = self._query_labevents(subject_ids, hadm_ids, lab_itemids)
            tables["labevents"] = self._build_labevents(raw_labs, subject_ids, hadm_ids)
        if "d_labitems" in requested:
            tables["d_labitems"] = self._query_d_labitems(lab_itemids)

        diagnoses = None
        if requested & {"diagnoses_icd", "d_icd_diagnoses"}:
            diagnoses = self._query_diagnoses(subject_ids, hadm_ids)
        if "diagnoses_icd" in requested:
            tables["diagnoses_icd"] = self._build_diagnoses(
                diagnoses,
                tables["admission_discharges"],
                subject_ids,
                hadm_ids,
            )
        if "d_icd_diagnoses" in requested:
            code_versions = _icd_code_versions(diagnoses)
            tables["d_icd_diagnoses"] = self._query_icd_dictionary(
                "d_icd_diagnoses", code_versions
            )

        if "prescriptions" in requested or "prescription_stops" in requested:
            raw_prescriptions = self._query_prescriptions(subject_ids, hadm_ids)
            prescriptions, stops = self._build_prescriptions(
                raw_prescriptions, subject_ids, hadm_ids
            )
            tables["prescriptions"] = prescriptions
            tables["prescription_stops"] = stops

        procedures = None
        if requested & {"procedures_icd", "d_icd_procedures"}:
            procedures = self._query_procedures(subject_ids, hadm_ids)
        if "procedures_icd" in requested:
            tables["procedures_icd"] = self._build_procedures(
                procedures, subject_ids, hadm_ids
            )
        if "d_icd_procedures" in requested:
            code_versions = _icd_code_versions(procedures)
            tables["d_icd_procedures"] = self._query_icd_dictionary(
                "d_icd_procedures", code_versions
            )

        if "drgcodes" in requested:
            raw_drgcodes = self._query_drgcodes(subject_ids, hadm_ids)
            tables["drgcodes"] = self._build_drgcodes(
                raw_drgcodes,
                tables["admission_discharges"],
                subject_ids,
                hadm_ids,
            )

        if "omr" in requested:
            raw_omr = self._query_omr(subject_ids)
            tables["omr"] = self._build_omr(raw_omr, subject_ids)
        return query_manifest

    def _get_client(self) -> None:
        if self.client is not None:
            return
        if not self.project_id:
            raise ValueError(
                "Subset needs BigQuery for hosp heavy/dictionary tables. "
                "Set PROJECT_ID or pass project_id explicitly."
            )
        from google.cloud import bigquery

        self.bigquery = bigquery
        self.client = bigquery.Client(
            project=self.project_id, location=self.config.location
        )

    def _run_query(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        self._get_client()
        job_config = self.bigquery.QueryJobConfig(query_parameters=params or [])
        return self.client.query_and_wait(query, job_config=job_config).to_dataframe()

    def _query_top_lab_itemids(
        self, subject_ids: list[int], hadm_ids: list[int]
    ) -> list[int]:
        if not subject_ids or not hadm_ids:
            return []
        self._get_client()
        limit_clause = "LIMIT @top_lab_items" if self.config.top_lab_items > 0 else ""
        query = f"""
        SELECT itemid
        FROM `{self.config.dataset_name}.{MIMIC_VERSION}_hosp.labevents`
        WHERE subject_id IN UNNEST(@subject_ids)
          AND hadm_id IN UNNEST(@hadm_ids)
          AND valuenum IS NOT NULL
          AND (storetime IS NOT NULL OR charttime IS NOT NULL)
        GROUP BY itemid
        ORDER BY COUNT(DISTINCT hadm_id) DESC, COUNT(*) DESC, itemid
        {limit_clause}
        """
        params = [
            _array_param(self.bigquery, "subject_ids", subject_ids, "INT64"),
            _array_param(self.bigquery, "hadm_ids", hadm_ids, "INT64"),
        ]
        if self.config.top_lab_items > 0:
            params.append(
                self.bigquery.ScalarQueryParameter(
                    "top_lab_items", "INT64", self.config.top_lab_items
                )
            )
        df = self._run_query(query, params)
        return _format_ids(df["itemid"])

    def _query_labevents(
        self, subject_ids: list[int], hadm_ids: list[int], itemids: list[int]
    ) -> pd.DataFrame:
        if not subject_ids or not hadm_ids or not itemids:
            return pd.DataFrame()
        self._get_client()
        query = f"""
        SELECT
            labevent_id,
            subject_id,
            hadm_id,
            itemid,
            storetime,
            charttime,
            valuenum,
            valueuom,
            ref_range_lower,
            ref_range_upper,
            flag,
            priority
        FROM `{self.config.dataset_name}.{MIMIC_VERSION}_hosp.labevents`
        WHERE subject_id IN UNNEST(@subject_ids)
          AND hadm_id IN UNNEST(@hadm_ids)
          AND itemid IN UNNEST(@itemids)
        ORDER BY subject_id, hadm_id, charttime, labevent_id
        """
        params = [
            _array_param(self.bigquery, "subject_ids", subject_ids, "INT64"),
            _array_param(self.bigquery, "hadm_ids", hadm_ids, "INT64"),
            _array_param(self.bigquery, "itemids", itemids, "INT64"),
        ]
        return self._run_query(query, params)

    def _query_d_labitems(self, itemids: list[int]) -> pd.DataFrame:
        if not itemids:
            return pd.DataFrame(columns=["itemid", "label", "fluid", "category"])
        self._get_client()
        query = f"""
        SELECT *
        FROM `{self.config.dataset_name}.{MIMIC_VERSION}_hosp.d_labitems`
        WHERE itemid IN UNNEST(@itemids)
        ORDER BY itemid
        """
        return self._run_query(
            query, [_array_param(self.bigquery, "itemids", itemids, "INT64")]
        )

    def _query_diagnoses(
        self, subject_ids: list[int], hadm_ids: list[int]
    ) -> pd.DataFrame:
        return self._query_hadm_table(
            "diagnoses_icd",
            subject_ids,
            hadm_ids,
            "subject_id, hadm_id, seq_num, icd_code, icd_version",
            "subject_id, hadm_id, seq_num",
        )

    def _query_prescriptions(
        self, subject_ids: list[int], hadm_ids: list[int]
    ) -> pd.DataFrame:
        return self._query_hadm_table(
            "prescriptions",
            subject_ids,
            hadm_ids,
            """
            subject_id,
            hadm_id,
            starttime,
            stoptime,
            drug_type,
            drug,
            dose_val_rx,
            dose_unit_rx,
            doses_per_24_hrs,
            route
            """,
            "subject_id, starttime, hadm_id",
        )

    def _query_procedures(
        self, subject_ids: list[int], hadm_ids: list[int]
    ) -> pd.DataFrame:
        return self._query_hadm_table(
            "procedures_icd",
            subject_ids,
            hadm_ids,
            "subject_id, hadm_id, seq_num, icd_code, icd_version, chartdate",
            "subject_id, hadm_id, seq_num",
        )

    def _query_drgcodes(
        self, subject_ids: list[int], hadm_ids: list[int]
    ) -> pd.DataFrame:
        return self._query_hadm_table(
            "drgcodes",
            subject_ids,
            hadm_ids,
            """
            subject_id,
            hadm_id,
            drg_type,
            drg_code,
            description,
            drg_severity,
            drg_mortality
            """,
            "subject_id, hadm_id, drg_type, drg_code",
        )

    def _query_omr(self, subject_ids: list[int]) -> pd.DataFrame:
        columns = ["subject_id", "chartdate", "seq_num", "result_name", "result_value"]
        if not subject_ids:
            return pd.DataFrame(columns=columns)
        self._get_client()
        query = f"""
        SELECT
            subject_id,
            chartdate,
            seq_num,
            result_name,
            result_value
        FROM `{self.config.dataset_name}.{MIMIC_VERSION}_hosp.omr`
        WHERE subject_id IN UNNEST(@subject_ids)
        ORDER BY subject_id, chartdate, seq_num
        """
        return self._run_query(
            query,
            [_array_param(self.bigquery, "subject_ids", subject_ids, "INT64")],
        )

    def _query_hadm_table(
        self,
        table_name: str,
        subject_ids: list[int],
        hadm_ids: list[int],
        select_cols: str,
        order_cols: str,
        extra_where: str = "",
    ) -> pd.DataFrame:
        if not subject_ids or not hadm_ids:
            return pd.DataFrame()
        self._get_client()
        query = f"""
        SELECT {select_cols}
        FROM `{self.config.dataset_name}.{MIMIC_VERSION}_hosp.{table_name}`
        WHERE subject_id IN UNNEST(@subject_ids)
          AND hadm_id IN UNNEST(@hadm_ids)
          {extra_where}
        ORDER BY {order_cols}
        """
        params = [
            _array_param(self.bigquery, "subject_ids", subject_ids, "INT64"),
            _array_param(self.bigquery, "hadm_ids", hadm_ids, "INT64"),
        ]
        return self._run_query(query, params)

    def _query_icd_dictionary(
        self, table_name: str, icd_code_versions: list[str]
    ) -> pd.DataFrame:
        columns = ["icd_code", "icd_version", "icd_code_version", "long_title"]
        if not icd_code_versions:
            return pd.DataFrame(columns=columns)
        self._get_client()
        query = f"""
        SELECT
            icd_code,
            icd_version,
            CONCAT(CAST(icd_version AS STRING), ':', icd_code) AS icd_code_version,
            long_title
        FROM `{self.config.dataset_name}.{MIMIC_VERSION}_hosp.{table_name}`
        WHERE CONCAT(CAST(icd_version AS STRING), ':', icd_code)
              IN UNNEST(@icd_code_versions)
        ORDER BY icd_version, icd_code
        """
        return self._run_query(
            query,
            [
                _array_param(
                    self.bigquery, "icd_code_versions", icd_code_versions, "STRING"
                )
            ],
        )

    def _build_labevents(
        self, labevents: pd.DataFrame, subject_ids: list[int], hadm_ids: list[int]
    ) -> pd.DataFrame:
        keep = [
            "labevent_id",
            "subject_id",
            "hadm_id",
            "itemid",
            "lab_time",
            "valuenum",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper",
            "flag",
            "priority",
        ]
        if labevents.empty or not {"subject_id", "hadm_id"}.issubset(labevents.columns):
            return pd.DataFrame(columns=keep)
        df = labevents[
            labevents["subject_id"].isin(subject_ids)
            & labevents["hadm_id"].isin(hadm_ids)
        ].copy()
        store = (
            parse_datetime(df["storetime"])
            if "storetime" in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        chart = (
            parse_datetime(df["charttime"])
            if "charttime" in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        df["lab_time"] = store.fillna(
            chart
        )  # storetime: the measurement was made available in the lab system -> can be used by model
        if self.config.require_numeric_labs and "valuenum" in df.columns:
            df = df[pd.to_numeric(df["valuenum"], errors="coerce").notna()]
        df = df[df["lab_time"].notna()]
        return _sort_existing(
            _select_existing(df, keep), ["subject_id", "lab_time", "labevent_id"]
        )

    def _build_diagnoses(
        self,
        diagnoses: pd.DataFrame,
        discharges: pd.DataFrame,
        subject_ids: list[int],
        hadm_ids: list[int],
    ) -> pd.DataFrame:
        keep = [
            "subject_id",
            "hadm_id",
            "seq_num",
            "icd_code",
            "icd_version",
            "icd_code_version",
            "diagnosis_time",
        ]
        if diagnoses is None or diagnoses.empty or "hadm_id" not in diagnoses.columns:
            return pd.DataFrame(columns=keep)
        df = diagnoses[diagnoses["hadm_id"].isin(hadm_ids)].copy()
        if "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subject_ids)]
        df = _with_icd_code_version(df)
        df = df.merge(discharges[["hadm_id", "dischtime"]], on="hadm_id", how="left")
        df = df.rename(columns={"dischtime": "diagnosis_time"})
        df = df[df["diagnosis_time"].notna()]
        return _sort_existing(
            _select_existing(df, keep),
            ["subject_id", "diagnosis_time", "hadm_id", "seq_num"],
        )

    def _build_prescriptions(
        self, prescriptions: pd.DataFrame, subject_ids: list[int], hadm_ids: list[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        start_keep = [
            "subject_id",
            "hadm_id",
            "starttime",
            "drug_type",
            "drug",
            "dose_val_rx",
            "dose_unit_rx",
            "doses_per_24_hrs",
            "route",
        ]
        stop_keep = ["subject_id", "hadm_id", "stoptime", "drug"]
        if prescriptions.empty or "hadm_id" not in prescriptions.columns:
            return pd.DataFrame(columns=start_keep), pd.DataFrame(columns=stop_keep)
        df = prescriptions[prescriptions["hadm_id"].isin(hadm_ids)].copy()
        if "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subject_ids)]
        for time_col in ("starttime", "stoptime"):
            if time_col in df.columns:
                df[time_col] = parse_datetime(df[time_col])
        stops = _select_existing(df, stop_keep)
        if self.config.require_prescription_stoptime and "stoptime" in stops.columns:
            stops = stops[stops["stoptime"].notna()]
        starts = df.drop(columns=["stoptime"], errors="ignore")
        if self.config.require_prescription_starttime and "starttime" in starts.columns:
            starts = starts[starts["starttime"].notna()]
        return (
            _sort_existing(
                _select_existing(starts, start_keep),
                ["subject_id", "starttime", "hadm_id"],
            ),
            _sort_existing(stops, ["subject_id", "stoptime", "hadm_id"]),
        )

    def _build_procedures(
        self, procedures: pd.DataFrame, subject_ids: list[int], hadm_ids: list[int]
    ) -> pd.DataFrame:
        keep = [
            "subject_id",
            "hadm_id",
            "seq_num",
            "icd_code",
            "icd_version",
            "icd_code_version",
            "procedure_time",
        ]
        if (
            procedures is None
            or procedures.empty
            or "hadm_id" not in procedures.columns
        ):
            return pd.DataFrame(columns=keep)
        df = procedures[procedures["hadm_id"].isin(hadm_ids)].copy()
        if "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subject_ids)]
        df = _with_icd_code_version(df)
        if "chartdate" in df.columns:
            df["procedure_time"] = parse_datetime(df["chartdate"])
        df = (
            df[df["procedure_time"].notna()]
            if "procedure_time" in df.columns
            else df.iloc[0:0]
        )
        return _sort_existing(
            _select_existing(df, keep),
            ["subject_id", "procedure_time", "hadm_id", "seq_num"],
        )

    def _build_drgcodes(
        self,
        drgcodes: pd.DataFrame,
        discharges: pd.DataFrame,
        subject_ids: list[int],
        hadm_ids: list[int],
    ) -> pd.DataFrame:
        keep = [
            "subject_id",
            "hadm_id",
            "drg_type",
            "drg_code",
            "description",
            "drg_severity",
            "drg_mortality",
            "drg_time",
        ]
        if drgcodes.empty or "hadm_id" not in drgcodes.columns:
            return pd.DataFrame(columns=keep)
        df = drgcodes[drgcodes["hadm_id"].isin(hadm_ids)].copy()
        if "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subject_ids)]
        df = df.merge(discharges[["hadm_id", "dischtime"]], on="hadm_id", how="left")
        df = df.rename(columns={"dischtime": "drg_time"})
        df = df[df["drg_time"].notna()]
        return _sort_existing(
            _select_existing(df, keep),
            ["subject_id", "drg_time", "hadm_id", "drg_type", "drg_code"],
        )

    def _build_omr(self, omr: pd.DataFrame, subject_ids: list[int]) -> pd.DataFrame:
        keep = ["subject_id", "omr_time", "seq_num", "result_name", "result_value"]
        if omr.empty or not {"subject_id", "chartdate"}.issubset(omr.columns):
            return pd.DataFrame(columns=keep)
        df = omr[omr["subject_id"].isin(subject_ids)].copy()
        df["omr_time"] = parse_datetime(df["chartdate"])
        df = df[df["omr_time"].notna()]
        return _sort_existing(
            _select_existing(df, keep),
            ["subject_id", "omr_time", "seq_num"],
        )

    def _save_cohort_ids(self, tables: dict[str, pd.DataFrame]) -> None:
        if "patients" in tables and "subject_id" in tables["patients"].columns:
            save_table(
                tables["patients"][["subject_id"]],
                self.config.out_dir / "cohort" / "subject_ids.parquet",
            )
        if "admissions" in tables and "hadm_id" in tables["admissions"].columns:
            save_table(
                tables["admissions"][["hadm_id"]],
                self.config.out_dir / "cohort" / "hadm_ids.parquet",
            )

    def _cohort_summary(self, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
        summary = {}
        if "patients" in tables:
            summary["patients"] = int(tables["patients"]["subject_id"].nunique())
        if "admissions" in tables:
            summary["admissions"] = int(tables["admissions"]["hadm_id"].nunique())
        return summary


def _select_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]].copy()


def _sort_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sort_cols = [col for col in columns if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.reset_index(drop=True)


def _with_icd_code_version(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"icd_version", "icd_code"}.issubset(
        df.columns
    ) and "icd_code_version" not in df.columns:
        df["icd_code_version"] = (
            df["icd_version"].astype("string") + ":" + df["icd_code"].astype("string")
        )
    return df


def _icd_code_versions(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return []
    df = _with_icd_code_version(df)
    if "icd_code_version" not in df.columns:
        return []
    return sorted(str(value) for value in df["icd_code_version"].dropna().unique())


def _format_ids(series: pd.Series) -> list[int]:
    return [int(value) for value in series.dropna().unique().tolist()]


def _array_param(bigquery: Any, name: str, values: list[Any], value_type: str) -> Any:
    return bigquery.ArrayQueryParameter(name, value_type, list(values))
