from __future__ import annotations

from typing import Dict, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd


from relbench.base import EntityTask
from relbench.base.database import Database
from relbench.base.table import Table


ALLOWED_ATTRIBUTE_FEATURE_VERSION = "allowed_attrs_fair_v1"


def _table(db: Database, name: str, columns: Iterable[str]) -> pd.DataFrame:
    table = db.table_dict.get(name)
    if table is None:
        return pd.DataFrame()
    keep = [column for column in columns if column in table.df.columns]
    return table.df[keep].copy()


def _empty_features(
    seeds: pd.DataFrame,
    columns: Iterable[str],
    zero_columns: Iterable[str] = (),
) -> pd.DataFrame:
    result = pd.DataFrame({"__row_id": seeds["__row_id"]})
    zero_columns = set(zero_columns)
    for column in columns:
        result[column] = 0.0 if column in zero_columns else np.nan
    return result


def _aggregate(
    seeds: pd.DataFrame,
    events: pd.DataFrame,
    time_col: str,
    expressions: Mapping[str, str],
    zero_columns: Iterable[str],
    inclusive: bool = True,
) -> pd.DataFrame:
    columns = list(expressions)
    if events.empty or not {"subject_id", time_col} <= set(events.columns):
        return _empty_features(seeds, columns, zero_columns)

    events = events.copy()
    events[time_col] = pd.to_datetime(events[time_col], errors="coerce")
    events = events[events[time_col].notna()]
    if events.empty:
        return _empty_features(seeds, columns, zero_columns)

    event_time_op = "<=" if inclusive else "<"
    select_sql = ",\n                ".join(
        f"{expression} AS {column}" for column, expression in expressions.items()
    )
    con = duckdb.connect()
    try:
        con.register("seeds", seeds[["__row_id", "subject_id", "seed_time"]])
        con.register("events", events)
        result = con.execute(
            f"""
            SELECT
                s.__row_id,
                {select_sql}
            FROM seeds s
            LEFT JOIN events e
              ON e.subject_id = s.subject_id
             AND e.{time_col} {event_time_op} s.seed_time
            GROUP BY s.__row_id
            ORDER BY s.__row_id
            """
        ).df()
    finally:
        con.close()

    for column in zero_columns:
        result[column] = result[column].fillna(0)
    return result


def _patient_features(db: Database, seeds: pd.DataFrame) -> pd.DataFrame:
    patients = _table(db, "patients", ("subject_id", "gender", "age", "anchor_age"))
    if patients.empty or "subject_id" not in patients.columns:
        return _empty_features(seeds, ("gender", "age"))

    result = seeds[["__row_id", "subject_id"]].merge(
        patients, on="subject_id", how="left"
    )
    if "age" in result.columns:
        result["age"] = pd.to_numeric(result["age"], errors="coerce")
    else:
        result["age"] = pd.to_numeric(result.get("anchor_age"), errors="coerce")
    if "gender" not in result.columns:
        result["gender"] = pd.NA
    return result[["__row_id", "gender", "age"]]


def _admission_features(db: Database, seeds: pd.DataFrame) -> pd.DataFrame:
    admissions = _table(db, "admissions", ("subject_id", "admittime"))
    expressions: Dict[str, str] = {
        "num_prev_admissions": "count(e.admittime)",
        "days_since_last_admission": (
            "date_diff('second', max(e.admittime), max(s.seed_time)) / 86400.0"
        ),
    }
    return _aggregate(
        seeds,
        admissions,
        "admittime",
        expressions,
        ("num_prev_admissions",),
        inclusive=False,
    )


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _string_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[column].astype("string")


def _lab_features(db: Database, seeds: pd.DataFrame) -> pd.DataFrame:
    labs = _table(
        db,
        "labevents",
        (
            "subject_id",
            "lab_time",
            "valuenum",
            "ref_range_lower",
            "ref_range_upper",
            "flag",
        ),
    )
    if not labs.empty:
        labs["valuenum"] = _numeric_column(labs, "valuenum")
        labs["ref_range_lower"] = _numeric_column(labs, "ref_range_lower")
        labs["ref_range_upper"] = _numeric_column(labs, "ref_range_upper")
        labs["flag"] = _string_column(labs, "flag")
        flag = labs["flag"].str.lower()
        labs["flag_abnormal"] = (flag.notna() & ~flag.isin(["", "normal"])).astype(
            "int8"
        )
        labs["below_ref"] = (
            labs["ref_range_lower"].notna()
            & labs["valuenum"].lt(labs["ref_range_lower"])
        ).astype("int8")
        labs["above_ref"] = (
            labs["ref_range_upper"].notna()
            & labs["valuenum"].gt(labs["ref_range_upper"])
        ).astype("int8")
        labs["is_abnormal"] = (
            labs["flag_abnormal"].astype(bool)
            | labs["below_ref"].astype(bool)
            | labs["above_ref"].astype(bool)
        ).astype("int8")

    expressions: Dict[str, str] = {
        "num_lab_events": "count(e.lab_time)",
        "lab_abnormal_ratio": "avg(e.is_abnormal)",
        "num_labs_last_24h": "count(e.lab_time) FILTER (WHERE e.lab_time >= s.seed_time - INTERVAL '1 day')",
        "num_labs_last_7d": "count(e.lab_time) FILTER (WHERE e.lab_time >= s.seed_time - INTERVAL '7 days')",
        "num_labs_last_30d": "count(e.lab_time) FILTER (WHERE e.lab_time >= s.seed_time - INTERVAL '30 days')",
    }
    counts = (
        "num_lab_events",
        "num_labs_last_24h",
        "num_labs_last_7d",
        "num_labs_last_30d",
    )
    return _aggregate(seeds, labs, "lab_time", expressions, counts)


def _prescription_features(db: Database, seeds: pd.DataFrame) -> pd.DataFrame:
    prescriptions = _table(db, "prescriptions", ("subject_id", "starttime", "drug"))
    if not prescriptions.empty:
        prescriptions["drug_norm"] = _string_column(prescriptions, "drug").str.lower()

    expressions: Dict[str, str] = {
        "num_prescriptions": "count(e.drug_norm)",
        "num_unique_drugs": "count(DISTINCT e.drug_norm)",
    }
    return _aggregate(seeds, prescriptions, "starttime", expressions, expressions)


def _prepare_icd_events(
    db: Database,
    table_name: str,
    time_col: str,
) -> pd.DataFrame:
    events = _table(
        db,
        table_name,
        ("subject_id", time_col, "icd_code", "icd_version"),
    )
    if events.empty:
        return events
    events["icd_code_clean"] = (
        _string_column(events, "icd_code")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    events["icd_version_num"] = _numeric_column(events, "icd_version")
    events["icd_key"] = (
        events["icd_version_num"].astype("Int64").astype("string")
        + ":"
        + events["icd_code_clean"]
    )
    return events


def _diagnosis_features(db: Database, seeds: pd.DataFrame) -> pd.DataFrame:
    diagnoses = _prepare_icd_events(db, "diagnoses_icd", "diagnosis_time")
    count_columns = [
        "num_prev_diagnoses",
        "num_unique_icd_codes",
        "prev_diagnosis_icd9_count",
        "prev_diagnosis_icd10_count",
    ]
    expressions: Dict[str, str] = {
        "num_prev_diagnoses": "count(e.icd_code_clean)",
        "num_unique_icd_codes": "count(DISTINCT e.icd_key)",
        "prev_diagnosis_icd9_count": "sum(CASE WHEN e.icd_version_num = 9 THEN 1 ELSE 0 END)",
        "prev_diagnosis_icd10_count": "sum(CASE WHEN e.icd_version_num = 10 THEN 1 ELSE 0 END)",
    }
    return _aggregate(seeds, diagnoses, "diagnosis_time", expressions, count_columns)


def _procedure_features(db: Database, seeds: pd.DataFrame) -> pd.DataFrame:
    procedures = _prepare_icd_events(db, "procedures_icd", "procedure_time")
    count_columns = [
        "num_prev_procedures",
        "num_unique_procedure_icd_codes",
        "prev_procedure_icd9_count",
        "prev_procedure_icd10_count",
    ]
    expressions: Dict[str, str] = {
        "num_prev_procedures": "count(e.icd_code_clean)",
        "num_unique_procedure_icd_codes": "count(DISTINCT e.icd_key)",
        "prev_procedure_icd9_count": "sum(CASE WHEN e.icd_version_num = 9 THEN 1 ELSE 0 END)",
        "prev_procedure_icd10_count": "sum(CASE WHEN e.icd_version_num = 10 THEN 1 ELSE 0 END)",
    }
    return _aggregate(seeds, procedures, "procedure_time", expressions, count_columns)


def build_allowed_attribute_features(
    db: Database,
    task: EntityTask,
    task_table: Table,
) -> pd.DataFrame:
    """Build point-in-time features from the approved MIMIC attributes only."""
    entity_col = next(iter(task_table.fkey_col_to_pkey_table))
    source = task_table.df.reset_index(drop=True).copy()
    seeds = pd.DataFrame(
        {
            "__row_id": np.arange(len(source), dtype=np.int64),
            "subject_id": source[entity_col].to_numpy(),
            "seed_time": pd.to_datetime(source[task.time_col], errors="coerce"),
        }
    )
    if seeds["seed_time"].isna().any():
        raise ValueError(f"{task.time_col} contains missing or invalid timestamps.")

    result = pd.DataFrame({"__row_id": seeds["__row_id"]})
    builders = (
        _patient_features,
        _admission_features,
        _lab_features,
        _prescription_features,
        _procedure_features,
        _diagnosis_features,
    )
    for builder in builders:
        result = result.merge(builder(db, seeds), on="__row_id", how="left")

    if task.target_col in source:
        result[task.target_col] = source[task.target_col].to_numpy()
    return result.drop(columns="__row_id")
