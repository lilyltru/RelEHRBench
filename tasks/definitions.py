"""Tasks for the hosp-only MIMIC-IV dataset."""

import duckdb
import pandas as pd

from relbench.base import TaskType
from relbench.base.database import Database
from relbench.base.table import Table
from relbench.base.task_entity import EntityTask
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    multilabel_auprc_macro,
    multilabel_auroc_macro,
    multilabel_f1_macro,
    multilabel_auprc_micro,
    multilabel_auroc_micro,
    multilabel_f1_micro,
    roc_auc,
)

from .labels import (
    CARDIOVASCULAR_SUBTYPE_ORDER,
    CARDIOVASCULAR_SUBTYPE_PATTERNS,
    CCSR_CODE_ORDER,
    CCSR_CODE_REGEX,
    DIGESTIVE_SUBTYPE_ORDER,
    DIGESTIVE_SUBTYPE_PATTERNS,
    HIGH_RISK_LAB_LABELS,
)


def _filter_joint_task_rows(
    dataset,
    split: str,
    table: Table,
    horizon: pd.Timedelta | None = None,
) -> Table:
    """Keep task rows in-split with complete label observation."""

    if not getattr(dataset, "is_joint_split", False):
        return table

    prediction_time = pd.to_datetime(table.df[table.time_col], errors="coerce")
    if split == "train":
        observation_end = dataset.val_timestamp
        inclusive = False
        prediction_mask = prediction_time < dataset.val_timestamp
    elif split == "val":
        observation_end = dataset.test_timestamp
        inclusive = False
        prediction_mask = (prediction_time >= dataset.val_timestamp) & (
            prediction_time < dataset.test_timestamp
        )
    elif split == "test":
        observation_end = dataset.end_timestamp
        inclusive = True
        prediction_mask = (prediction_time >= dataset.test_timestamp) & (
            prediction_time <= observation_end
        )
    else:
        raise ValueError(f"Unknown split={split!r}")

    if pd.isna(observation_end):
        mask = pd.Series(False, index=table.df.index)
    else:
        mask = prediction_mask
        if horizon is not None:
            label_end = prediction_time + horizon
            if inclusive:
                mask &= label_end <= observation_end
            else:
                # The preceding cumulative DB excludes the cutoff itself.
                mask &= label_end < observation_end

    return Table(
        df=table.df.loc[mask].reset_index(drop=True).copy(),
        fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def _completed_admissions(db: Database) -> pd.DataFrame:
    """Return admissions whose actual discharge is in the same split DB."""

    admissions = db.table_dict["admissions"].df
    discharges = db.table_dict["admission_discharges"].df
    completed_hadm_ids = discharges.loc[
        discharges["dischtime"].notna(), "hadm_id"
    ].dropna()
    return admissions[admissions["hadm_id"].isin(completed_hadm_ids)].copy()


class HighRiskLabAbnormalPatientSplitTask(EntityTask):
    r"""Predict a high-risk abnormal lab on an admission-relative 24h grid.

    Each admission ``a`` contributes prediction times
    ``t[a, k] = admittime(a) + k * 24h`` for ``k = 0, 1, ...``.  The seed
    entity remains the patient, while ``hadm_id`` identifies the admission
    whose next-24-hour lab window is labeled.  A row is retained only when the
    complete label window is observed before the admission/discharge boundary
    and, for a joint split.
    """

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "subject_id"
    entity_table = "patients"
    time_col = "timestamp"
    target_col = "high_risk_lab_abnormal_24h"
    timedelta = pd.Timedelta(hours=24)
    num_eval_timestamps = 1  # unused; admission-relative times are generated below
    metrics = [average_precision, accuracy, f1, roc_auc]
    high_risk_lab_labels = HIGH_RISK_LAB_LABELS

    @staticmethod
    def _normalize_lab_label(label: object) -> str:
        return " ".join(str(label).strip().lower().split())

    def _lab_itemids_for_db(self, db: Database) -> tuple[int, ...]:
        if "d_labitems" not in db.table_dict:
            raise RuntimeError(
                "HighRiskLabAbnormalPatientSplitTask requires d_labitems so "
                "lab labels can be resolved in the split DB's itemid space."
            )

        d_labitems = db.table_dict["d_labitems"].df
        if not {"itemid", "label"} <= set(d_labitems.columns):
            raise RuntimeError("d_labitems must contain itemid and label columns.")

        target_labels = {
            self._normalize_lab_label(label) for label in self.high_risk_lab_labels
        }
        label_mask = (
            d_labitems["label"].map(self._normalize_lab_label).isin(target_labels)
        )
        itemids = d_labitems.loc[label_mask, "itemid"].dropna().astype(int).unique()
        if len(itemids) == 0:
            raise RuntimeError(
                "None of the configured high-risk lab labels were found in "
                "d_labitems. Check the lab subset/profile used to build this cache."
            )
        return tuple(sorted(itemids))

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        raise NotImplementedError(
            "HighRiskLabAbnormalPatientSplitTask uses admission-relative "
            "prediction times and overrides _get_table()."
        )

    def _build_table_from_db(
        self,
        db: Database,
        observation_end: pd.Timestamp,
        allow_truncated_discharges: bool = False,
    ) -> Table:
        admissions = db.table_dict["admissions"].df
        admission_discharges = db.table_dict["admission_discharges"].df
        labevents = db.table_dict["labevents"].df
        duckdb.register("admissions", admissions)
        duckdb.register("admission_discharges", admission_discharges)
        duckdb.register("labevents", labevents)
        lab_itemids = self._lab_itemids_for_db(db)
        itemids_sql = ", ".join(str(itemid) for itemid in lab_itemids)
        observation_end = pd.Timestamp(observation_end)
        truncated_discharge_predicate = (
            "TRUE" if allow_truncated_discharges else "d.dischtime IS NOT NULL"
        )

        # Our PyHealth lab task skips admissions without a recorded discharge.
        # Our RelBench lab task uses the split end if the joint split lacks it.
        # Both rules agree on eligibility for the dataset from our current pipeline.
        # TODO: Align discharge handling to support other pipelines that retain 
        # admissions without a recorded discharge for more robustness.
        df = duckdb.sql(
            f"""
            WITH admission_windows AS (
                SELECT
                    a.hadm_id,
                    a.subject_id,
                    a.admittime,
                    COALESCE(
                        d.dischtime,
                        TIMESTAMP '{observation_end.isoformat()}'
                    ) AS label_observation_end
                FROM admissions a
                LEFT JOIN admission_discharges d
                    ON d.hadm_id = a.hadm_id
                WHERE a.admittime IS NOT NULL
                  AND {truncated_discharge_predicate}
            ),
            prediction_times AS (
                SELECT
                    w.hadm_id,
                    w.subject_id,
                    grid.timestamp
                FROM admission_windows w
                CROSS JOIN LATERAL generate_series(
                    w.admittime,
                    w.label_observation_end - INTERVAL '{self.timedelta}',
                    INTERVAL '{self.timedelta}'
                ) AS grid(timestamp)
                WHERE w.admittime + INTERVAL '{self.timedelta}'
                      <= w.label_observation_end
            )
            SELECT
                p.timestamp AS {self.time_col},
                p.subject_id AS {self.entity_col},
                p.hadm_id,
                CAST(
                    EXISTS (
                        SELECT 1
                        FROM labevents l
                        WHERE l.hadm_id = p.hadm_id
                          AND l.itemid IN ({itemids_sql})
                          AND l.flag = 'abnormal'
                          AND l.lab_time > p.timestamp
                          AND l.lab_time <= p.timestamp
                                                + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS {self.target_col}
            FROM prediction_times p
            """
        ).df()

        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce").astype(
            "datetime64[ns]"
        )
        df = (
            df.dropna(subset=[self.time_col])
            .sort_values([self.time_col, self.entity_col, "hadm_id"])
            .reset_index(drop=True)
        )

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
                "hadm_id": "admissions",
            },
            pkey_col=None,
            time_col=self.time_col,
        )

    def _get_table(self, split: str) -> Table:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split={split!r}")
        if not hasattr(self.dataset, "get_split_db"):
            raise ValueError(
                "HighRiskLabAbnormalPatientSplitTask expects a dataset with "
                "patient-level get_split_db()."
            )

        split_db = self.dataset.get_split_db(split, upto_test_timestamp=False)
        is_joint_split = getattr(self.dataset, "is_joint_split", False)
        if not is_joint_split:
            # Patient split has no temporal cutoff. Complete label windows are
            # determined by each admission's discharge time.
            observation_end = split_db.max_timestamp
        elif split == "train":
            observation_end = self.dataset.val_timestamp
        elif split == "val":
            observation_end = self.dataset.test_timestamp
        else:
            observation_end = self.dataset.end_timestamp

        table = self._build_table_from_db(
            split_db,
            observation_end,
            allow_truncated_discharges=is_joint_split,
        )
        table = _filter_joint_task_rows(self.dataset, split, table, self.timedelta)
        return self.filter_dangling_entities(table)


class PatientHospitalAdmission30DPatientSplitTask(EntityTask):
    r"""30-day hospital readmission anchored at every discharge, with a
    patient-level (cohort) split.

    Splits are drawn at the patient (``subject_id``) level via
    :meth:`MimicPatientSplitDataset.get_patient_split`. For each split, the underlying
    Database is filtered to that cohort
    (:meth:`MimicPatientSplitDataset.get_split_db`). 
    The label is whether the patient has any new hospital
    admission strictly after ``dischtime`` and within 30 days.
    Drop last admission_discharges.
    """

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "subject_id"
    entity_table = "patients"
    time_col = "timestamp"
    target_col = "hosp_readmit_30d"
    timedelta = pd.Timedelta(days=30)
    num_eval_timestamps = 1  # unused (we override _get_table)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        # Not used by this task; _get_table is overridden. Implemented for
        # interface completeness.
        raise NotImplementedError(
            "PatientHospitalAdmission30DPatientSplitTask uses _get_table "
            "directly; make_table is not called."
        )

    def _build_table_from_db(self, db: Database) -> Table:
        admissions = db.table_dict["admissions"].df
        admission_discharges = db.table_dict["admission_discharges"].df
        duckdb.register("admissions", admissions)
        duckdb.register("admission_discharges", admission_discharges)

        df = duckdb.sql(
            f"""
            SELECT
                d.dischtime                       AS {self.time_col},
                d.subject_id                      AS {self.entity_col},
                CAST(
                    EXISTS (
                        SELECT 1
                        FROM admissions a_next
                        WHERE a_next.subject_id = d.subject_id
                          AND a_next.admittime  >  d.dischtime
                          AND a_next.admittime <= d.dischtime
                                                 + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                )                                  AS {self.target_col}
            FROM admission_discharges d
            WHERE d.dischtime IS NOT NULL
            """
        ).df()

        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce").astype(
            "datetime64[ns]"
        )
        df = df.dropna(subset=[self.time_col]).reset_index(drop=True)
        if not getattr(self.dataset, "is_joint_split", False):
            df = df.sort_values(
                [self.entity_col, self.time_col], ascending=[True, False]
            )
            df["__rn_desc"] = df.groupby(self.entity_col).cumcount() + 1
            df = df[df["__rn_desc"] > 1].drop(columns="__rn_desc")
            df = df.reset_index(drop=True)

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

    def _get_table(self, split: str) -> Table:
        split_db = self.dataset.get_split_db(split, upto_test_timestamp=False)
        table = self._build_table_from_db(split_db)
        table = _filter_joint_task_rows(self.dataset, split, table, self.timedelta)
        table = self.filter_dangling_entities(table)
        return table


class _AdmissionDiagnosisMultilabelTask(EntityTask):
    """Shared admission-time label construction for ICD-based tasks."""

    task_type = TaskType.MULTILABEL_CLASSIFICATION
    entity_col = "subject_id"
    entity_table = "patients"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=1)
    metrics = [
        multilabel_auprc_macro,
        multilabel_auroc_macro,
        multilabel_f1_macro,
        multilabel_auprc_micro,
        multilabel_auroc_micro,
        multilabel_f1_micro,
    ]
    patterns: dict[str, dict[str, str]]
    subtype_order: tuple[str, ...]
    keep_only_labeled = False

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]"):
        raise NotImplementedError("This task builds split tables directly.")

    def _build_table_from_db(self, db: Database) -> Table:
        admissions_source = (
            _completed_admissions(db)
            if getattr(self.dataset, "is_joint_split", False)
            else db.table_dict["admissions"].df
        )
        admissions = admissions_source[["hadm_id", "subject_id", "admittime"]].copy()
        diagnoses = (
            db.table_dict["diagnoses_icd"]
            .df[["hadm_id", "icd_version", "icd_code"]]
            .dropna(subset=["hadm_id", "icd_version", "icd_code"])
        )
        diagnoses = diagnoses.copy()
        diagnoses["icd_version"] = pd.to_numeric(
            diagnoses["icd_version"], errors="coerce"
        )
        diagnoses["icd_code"] = diagnoses["icd_code"].astype("string").str.upper()

        labels = pd.DataFrame({"hadm_id": admissions["hadm_id"].unique()})
        for subtype in self.subtype_order:
            pattern = self.patterns[subtype]
            matched = (
                diagnoses["icd_version"].eq(10)
                & diagnoses["icd_code"].str.match(pattern["icd10"], na=False)
            ) | (
                diagnoses["icd_version"].eq(9)
                & diagnoses["icd_code"].str.match(pattern["icd9"], na=False)
            )
            matched_hadm_ids = diagnoses.loc[matched, "hadm_id"].unique()
            labels[subtype] = labels["hadm_id"].isin(matched_hadm_ids).astype("int64")

        matrix = labels[list(self.subtype_order)].to_numpy(dtype="int64")
        if self.keep_only_labeled:
            labels = labels.loc[matrix.sum(axis=1) > 0].reset_index(drop=True)
            matrix = labels[list(self.subtype_order)].to_numpy(dtype="int64")
        labels[self.target_col] = [matrix[index] for index in range(len(matrix))]

        df = admissions.merge(
            labels[["hadm_id", self.target_col]], on="hadm_id", how="inner"
        ).rename(columns={"admittime": self.time_col})
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        df = (
            df.dropna(subset=[self.time_col])
            .sort_values([self.time_col, self.entity_col, "hadm_id"])[
                [self.time_col, self.entity_col, "hadm_id", self.target_col]
            ]
            .reset_index(drop=True)
        )
        if df.empty:
            raise ValueError(f"No samples found for {self.target_col}")
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
                "hadm_id": "admissions",
            },
            pkey_col=None,
            time_col=self.time_col,
        )

    def _get_table(self, split: str) -> Table:
        split_db = self.dataset.get_split_db(split, upto_test_timestamp=False)
        table = self._build_table_from_db(split_db)
        table = _filter_joint_task_rows(self.dataset, split, table)
        return self.filter_dangling_entities(table)


class AdmissionDigestiveSubtypeMultilabelTask(_AdmissionDiagnosisMultilabelTask):
    target_col = "digestive_subtype_multihot"
    patterns = DIGESTIVE_SUBTYPE_PATTERNS
    subtype_order = DIGESTIVE_SUBTYPE_ORDER
    num_labels = len(subtype_order)
    keep_only_labeled = True


class AdmissionCardiovascularSubtypeMultilabelTask(_AdmissionDiagnosisMultilabelTask):
    target_col = "cardiovascular_subtype_multihot"
    patterns = CARDIOVASCULAR_SUBTYPE_PATTERNS
    subtype_order = CARDIOVASCULAR_SUBTYPE_ORDER
    num_labels = len(subtype_order)
    keep_only_labeled = True


class AdmissionCCSRMultilabelTask(_AdmissionDiagnosisMultilabelTask):
    target_col = "ccsr_multihot"
    patterns = CCSR_CODE_REGEX
    subtype_order = CCSR_CODE_ORDER
    num_labels = len(subtype_order)
