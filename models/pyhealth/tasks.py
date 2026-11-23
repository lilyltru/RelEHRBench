from __future__ import annotations

import re
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Type

from pyhealth.tasks.base_task import BaseTask  # type: ignore[reportMissingImports]

from tasks.labels import (
    CARDIOVASCULAR_SUBTYPE_ORDER,
    CARDIOVASCULAR_SUBTYPE_PATTERNS,
    CCSR_CODE_ORDER,
    CCSR_CODE_REGEX,
    DIGESTIVE_SUBTYPE_ORDER,
    DIGESTIVE_SUBTYPE_PATTERNS,
    HIGH_RISK_LAB_LABELS,
)


class MimicTokenConstruction:
    """Reusable token construction for MIMIC-Hosp StageNet tasks."""

    @staticmethod
    def _raw_attr(event: Any, name: str) -> Any:
        return event.attr_dict.get(name)

    @staticmethod
    def _text_attr(event: Any, name: str) -> Optional[str]:
        value = event.attr_dict.get(name)
        if value is None:
            return None
        value = str(value).strip()
        if value == "" or value.lower() in {"nan", "none", "<na>"}:
            return None
        return value

    @staticmethod
    def _token_text(value: str) -> str:
        return value.strip().replace(" ", "_")

    @classmethod
    def _float_attr(cls, event: Any, name: str) -> Optional[float]:
        value = cls._text_attr(event, name)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _age_bin(age: float) -> str:
        if age < 18:
            return "0_17"
        if age < 30:
            return "18_29"
        if age < 45:
            return "30_44"
        if age < 60:
            return "45_59"
        if age < 75:
            return "60_74"
        if age < 90:
            return "75_89"
        return "90_PLUS"

    def _patient_tokens(self, patient: Any) -> List[str]:
        tokens: List[str] = []
        patient_events = patient.get_events(event_type="patients")
        if not patient_events:
            return tokens

        patient_event = patient_events[0]
        gender = self._text_attr(patient_event, "gender")
        age = self._float_attr(patient_event, "anchor_age")

        if gender is not None:
            tokens.append(f"GENDER:{self._token_text(gender)}")
        if age is not None:
            tokens.append(f"AGE_BIN:{self._age_bin(age)}")

        return tokens

    def _lab_result_token(self, event: Any) -> Optional[str]:
        itemid = self._text_attr(event, "itemid")
        if itemid is None:
            return None

        flag = self._text_attr(event, "flag")
        value = self._float_attr(event, "valuenum")
        lower = self._float_attr(event, "ref_range_lower")
        upper = self._float_attr(event, "ref_range_upper")

        if flag is not None:
            status = f"FLAG_{self._token_text(flag).upper()}"
        elif value is None:
            status = "OBSERVED"
        elif lower is not None and value < lower:
            status = "LOW"
        elif upper is not None and value > upper:
            status = "HIGH"
        elif lower is not None or upper is not None:
            status = "NORMAL"
        else:
            status = "NUMERIC"

        return f"LAB:{self._token_text(itemid)}:{status}"

    def _get_visit_codes(
        self,
        patient: Any,
        hadm_id: Any,
        cutoff_time,
    ) -> List[str]:
        diagnoses = patient.get_events(
            event_type="diagnoses_icd",
            end=cutoff_time,
            filters=[("hadm_id", "==", hadm_id)],
        )
        procedures = patient.get_events(
            event_type="procedures_icd",
            end=cutoff_time,
            filters=[("hadm_id", "==", hadm_id)],
        )
        prescriptions = patient.get_events(
            event_type="prescriptions",
            end=cutoff_time,
            filters=[("hadm_id", "==", hadm_id)],
        )
        labs = patient.get_events(
            event_type="labevents",
            end=cutoff_time,
            filters=[("hadm_id", "==", hadm_id)],
        )

        tokens: List[str] = []

        for event in diagnoses:
            code = self._text_attr(event, "icd_code")
            version = self._text_attr(event, "icd_version") or "UNK"
            if code is not None:
                tokens.append(
                    f"DX:{self._token_text(version)}:{self._token_text(code)}"
                )

        for event in procedures:
            code = self._text_attr(event, "icd_code")
            version = self._text_attr(event, "icd_version") or "UNK"
            if code is not None:
                tokens.append(
                    f"PX:{self._token_text(version)}:{self._token_text(code)}"
                )

        for event in prescriptions:
            drug = self._text_attr(event, "drug")
            if drug is not None:
                tokens.append(f"RX:{self._token_text(drug)}")

        for event in labs:
            lab_token = self._lab_result_token(event)
            if lab_token is not None:
                tokens.append(lab_token)

        return sorted(set(tokens))

    def _historical_stage_codes(
        self,
        patient: Any,
        admissions: Iterable[Any],
        cutoff_time,
    ) -> tuple[list[float], list[list[str]]]:
        patient_tokens = self._patient_tokens(patient)
        visit_codes: List[List[str]] = []
        visit_timestamps = []

        for admission in admissions:
            hadm_id = self._raw_attr(admission, "hadm_id")
            codes = self._get_visit_codes(
                patient=patient,
                hadm_id=hadm_id,
                cutoff_time=cutoff_time,
            )

            if not visit_codes and patient_tokens:
                codes = sorted(set(codes + patient_tokens))

            if not codes:
                continue

            visit_codes.append(codes)
            visit_timestamps.append(admission.timestamp)

        visit_intervals = [0.0]
        for previous_time, current_time in zip(
            visit_timestamps[:-1], visit_timestamps[1:]
        ):
            hours = (current_time - previous_time).total_seconds() / 3600.0
            visit_intervals.append(max(hours, 0.0))

        return visit_intervals, visit_codes

    @classmethod
    def _normalize_icd_version(cls, event: Any) -> Optional[str]:
        version = cls._text_attr(event, "icd_version")
        if version is None:
            return None
        return version.split(".", 1)[0]

    @classmethod
    def _diagnosis_multilabels(
        cls,
        diagnosis_events: Iterable[Any],
        patterns: dict[str, dict[str, str]],
        label_order: Iterable[str],
    ) -> list[str]:
        """
        Maps diagnosis events to an ordered set of disease labels.
        Duplicate matches are removed, and the returned labels follow the order specified by ``label_order``.
        """
        active_labels = set()

        for event in diagnosis_events:
            icd_version = cls._normalize_icd_version(event)
            icd_code = cls._text_attr(event, "icd_code")
            if icd_version is None or icd_code is None:
                continue

            icd_code = icd_code.upper()
            pattern_key = (
                "icd10"
                if icd_version == "10"
                else "icd9" if icd_version == "9" else None
            )
            if pattern_key is None:
                continue

            for label in label_order:
                if re.match(patterns[label][pattern_key], icd_code):
                    active_labels.add(label)

        return [label for label in label_order if label in active_labels]


class Readmission30DStageNetTask(MimicTokenConstruction, BaseTask):
    """One sample per non-final discharge for 30-day readmission."""

    task_name = "readmission_30d_stagenet_v2"
    label_key = "readmission"
    task_type = "binary"
    include_last_discharge = False

    def __init__(self, padding: int = 5):
        self.horizon = timedelta(days=30)
        self.padding = padding
        self.input_schema = {
            "codes": ("stagenet", {"padding": padding}),
        }
        self.output_schema = {
            self.label_key: "binary",
        }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        admissions = [
            event
            for event in patient.get_events(event_type="admissions")
            if event.timestamp is not None
            and self._raw_attr(event, "hadm_id") is not None
        ]
        discharges = [
            event
            for event in patient.get_events(event_type="admission_discharges")
            if event.timestamp is not None
            and self._raw_attr(event, "hadm_id") is not None
        ]

        admissions.sort(
            key=lambda event: (event.timestamp, str(self._raw_attr(event, "hadm_id")))
        )
        discharges.sort(
            key=lambda event: (event.timestamp, str(self._raw_attr(event, "hadm_id")))
        )

        anchor_discharges = (
            discharges if self.include_last_discharge else discharges[:-1]
        )
        if not anchor_discharges:
            return []

        samples: List[Dict[str, Any]] = []
        for discharge in anchor_discharges:
            anchor_time = discharge.timestamp
            anchor_hadm_id = self._raw_attr(discharge, "hadm_id")
            window_end = anchor_time + self.horizon

            label = int(
                any(
                    anchor_time < admission.timestamp <= window_end
                    for admission in admissions
                )
            )

            historical_admissions = [
                admission
                for admission in admissions
                if admission.timestamp <= anchor_time
            ]
            visit_intervals, visit_codes = self._historical_stage_codes(
                patient=patient,
                admissions=historical_admissions,
                cutoff_time=anchor_time,
            )

            if not visit_codes:
                continue

            samples.append(
                {
                    "patient_id": str(patient.patient_id),
                    "visit_id": str(anchor_hadm_id),
                    "timestamp": anchor_time.isoformat(),
                    "codes": (visit_intervals, visit_codes),
                    self.label_key: label,
                }
            )

        return samples


class HighRiskLabAbnormal24HStageNetTask(MimicTokenConstruction, BaseTask):
    """Predict an abnormal high-risk lab in the next 24 hours.

    This mirrors ``HighRiskLabAbnormalPatientSplitTask`` in RelBench: every
    completed admission contributes admission-relative 24-hour prediction
    times, and only complete windows ending on or before discharge are kept.
    The PyHealth input remains the existing StageNet feature representation
    built by :class:`MimicTokenConstruction` and is cut off at prediction time.

    As in RelBench, all item IDs whose normalized ``d_labitems.label`` matches
    one of the nine configured labels are used for target construction.
    """

    task_name = "high_risk_lab_24h_stagenet_v1"
    label_key = "high_risk_lab_abnormal_24h"
    task_type = "binary"
    high_risk_lab_labels = HIGH_RISK_LAB_LABELS

    def __init__(
        self,
        padding: int = 5,
        high_risk_lab_itemids: Iterable[int] | None = None,
    ):
        if high_risk_lab_itemids is None:
            # Import lazily so task definitions remain independent of the local
            # dataset adaptor when callers provide an explicit item-ID set.
            from .dataset import resolve_lab_itemids

            high_risk_lab_itemids = resolve_lab_itemids(self.high_risk_lab_labels)

        self.high_risk_lab_itemids = frozenset(
            int(itemid) for itemid in high_risk_lab_itemids
        )
        if not self.high_risk_lab_itemids:
            raise ValueError("high_risk_lab_itemids must contain at least one item ID")
        self.horizon = timedelta(hours=24)
        self.padding = padding
        self.input_schema = {
            "codes": ("stagenet", {"padding": padding}),
        }
        self.output_schema = {
            self.label_key: "binary",
        }

    def _is_high_risk_abnormal_lab(self, event: Any) -> bool:
        itemid = self._raw_attr(event, "itemid")
        try:
            itemid = int(float(itemid))
        except (TypeError, ValueError, OverflowError):
            return False

        flag = self._text_attr(event, "flag")
        return (
            itemid in self.high_risk_lab_itemids
            and flag is not None
            and flag.lower() == "abnormal"
        )

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        admissions = [
            event
            for event in patient.get_events(event_type="admissions")
            if event.timestamp is not None
            and self._raw_attr(event, "hadm_id") is not None
        ]
        discharges = [
            event
            for event in patient.get_events(event_type="admission_discharges")
            if event.timestamp is not None
            and self._raw_attr(event, "hadm_id") is not None
        ]
        admissions.sort(
            key=lambda event: (event.timestamp, str(self._raw_attr(event, "hadm_id")))
        )

        discharge_time_by_hadm_id = {
            self._raw_attr(discharge, "hadm_id"): discharge.timestamp
            for discharge in discharges
        }

        samples: List[Dict[str, Any]] = []
        for admission in admissions:
            anchor_hadm_id = self._raw_attr(admission, "hadm_id")
            discharge_time = discharge_time_by_hadm_id.get(anchor_hadm_id)
            if discharge_time is None:
                continue

            admission_labs = [
                event
                for event in patient.get_events(
                    event_type="labevents",
                    filters=[("hadm_id", "==", anchor_hadm_id)],
                )
                if event.timestamp is not None
                and self._is_high_risk_abnormal_lab(event)
            ]

            anchor_time = admission.timestamp
            while anchor_time + self.horizon <= discharge_time:
                window_end = anchor_time + self.horizon
                label = int(
                    any(
                        anchor_time < lab.timestamp <= window_end
                        for lab in admission_labs
                    )
                )

                historical_admissions = [
                    historical_admission
                    for historical_admission in admissions
                    if historical_admission.timestamp <= anchor_time
                ]
                visit_intervals, visit_codes = self._historical_stage_codes(
                    patient=patient,
                    admissions=historical_admissions,
                    cutoff_time=anchor_time,
                )

                if visit_codes:
                    samples.append(
                        {
                            "patient_id": str(patient.patient_id),
                            "visit_id": str(anchor_hadm_id),
                            "timestamp": anchor_time.isoformat(),
                            "codes": (visit_intervals, visit_codes),
                            self.label_key: label,
                        }
                    )

                anchor_time = window_end

        return samples


class _AdmissionDiagnosisStageNetTask(MimicTokenConstruction, BaseTask):
    """Shared admission-time sampling for ICD-based multilabel tasks."""

    task_type = "multilabel"
    patterns: dict[str, dict[str, str]]
    label_order: tuple[str, ...]
    require_label = False

    def __init__(self, padding: int = 5):
        self.padding = padding
        self.input_schema = {"codes": ("stagenet", {"padding": padding})}
        self.output_schema = {self.label_key: "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        admissions = [
            event
            for event in patient.get_events(event_type="admissions")
            if event.timestamp is not None
            and self._raw_attr(event, "hadm_id") is not None
        ]
        admissions.sort(
            key=lambda event: (
                event.timestamp,
                str(self._raw_attr(event, "hadm_id")),
            )
        )

        samples = []
        for admission in admissions:
            anchor_time = admission.timestamp
            hadm_id = self._raw_attr(admission, "hadm_id")
            labels = self._diagnosis_multilabels(
                diagnosis_events=patient.get_events(
                    event_type="diagnoses_icd",
                    filters=[("hadm_id", "==", hadm_id)],
                ),
                patterns=self.patterns,
                label_order=self.label_order,
            )
            if self.require_label and not labels:
                continue

            historical_admissions = [
                event for event in admissions if event.timestamp <= anchor_time
            ]
            visit_intervals, visit_codes = self._historical_stage_codes(
                patient=patient,
                admissions=historical_admissions,
                cutoff_time=anchor_time,
            )
            if visit_codes:
                samples.append(
                    {
                        "patient_id": str(patient.patient_id),
                        "visit_id": str(hadm_id),
                        "timestamp": anchor_time.isoformat(),
                        "codes": (visit_intervals, visit_codes),
                        self.label_key: labels,
                    }
                )
        return samples


class AdmissionDigestiveSubtypeStageNetTask(_AdmissionDiagnosisStageNetTask):
    task_name = "digestive_subtype_stagenet_v2"
    label_key = "digestive_subtype"
    patterns = DIGESTIVE_SUBTYPE_PATTERNS
    label_order = DIGESTIVE_SUBTYPE_ORDER
    require_label = True


class AdmissionCardiovascularSubtypeStageNetTask(_AdmissionDiagnosisStageNetTask):
    task_name = "cardiovascular_subtype_stagenet_v2"
    label_key = "cardiovascular_subtype"
    patterns = CARDIOVASCULAR_SUBTYPE_PATTERNS
    label_order = CARDIOVASCULAR_SUBTYPE_ORDER
    require_label = True


class AdmissionCCSRPhenotypingStageNetTask(_AdmissionDiagnosisStageNetTask):
    task_name = "phenotyping_stagenet_v2"
    label_key = "phenotyping"
    patterns = CCSR_CODE_REGEX
    label_order = CCSR_CODE_ORDER


TASK_REGISTRY: dict[str, Type[BaseTask]] = {
    "readmission_30d": Readmission30DStageNetTask,
    "high_risk_lab_24h": HighRiskLabAbnormal24HStageNetTask,
    "digestive_subtype": AdmissionDigestiveSubtypeStageNetTask,
    "cardiovascular_subtype": AdmissionCardiovascularSubtypeStageNetTask,
    "phenotyping": AdmissionCCSRPhenotypingStageNetTask,
}


def get_task(task_name: str, padding: int = 5) -> BaseTask:
    """Instantiate a supported PyHealth task."""
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task_name={task_name!r}. Available tasks: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_name](padding=padding)
