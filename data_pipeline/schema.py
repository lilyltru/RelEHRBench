from dataclasses import dataclass, field


MIMIC_VERSION = "mimiciv_2_2"


@dataclass(frozen=True)
class RawTableSpec:
    module: str
    table: str

    @property
    def dataset(self) -> str:
        return f"{MIMIC_VERSION}_{self.module}"


@dataclass(frozen=True)
class RelBenchTableSpec:
    name: str
    pkey_col: str | None
    time_col: str | None
    fkey_col_to_pkey_table: dict[str, str] = field(default_factory=dict)


RAW_TABLES: dict[str, RawTableSpec] = {
    "patients": RawTableSpec("hosp", "patients"),
    "admissions": RawTableSpec("hosp", "admissions"),
}

DEFAULT_RAW_TABLES = tuple(RAW_TABLES)


RELBENCH_TABLES: dict[str, RelBenchTableSpec] = {
    "patients": RelBenchTableSpec("patients", "subject_id", None),
    "admissions": RelBenchTableSpec(
        "admissions",
        "hadm_id",
        "admittime",
        {"subject_id": "patients"},
    ),
    "admission_discharges": RelBenchTableSpec(
        "admission_discharges",
        None,
        "dischtime",
        {"subject_id": "patients", "hadm_id": "admissions"},
    ),
    "d_labitems": RelBenchTableSpec("d_labitems", "itemid", None),
    "labevents": RelBenchTableSpec(
        "labevents",
        "labevent_id",
        "lab_time",
        {
            "subject_id": "patients",
            "hadm_id": "admissions",
            "itemid": "d_labitems",
        },
    ),
    "d_icd_diagnoses": RelBenchTableSpec("d_icd_diagnoses", "icd_code_version", None),
    "diagnoses_icd": RelBenchTableSpec(
        "diagnoses_icd",
        None,
        "diagnosis_time",
        {
            "subject_id": "patients",
            "hadm_id": "admissions",
            "icd_code_version": "d_icd_diagnoses",
        },
    ),
    "prescriptions": RelBenchTableSpec(
        "prescriptions",
        None,
        "starttime",
        {"subject_id": "patients", "hadm_id": "admissions"},
    ),
    "prescription_stops": RelBenchTableSpec(
        "prescription_stops",
        None,
        "stoptime",
        {"subject_id": "patients", "hadm_id": "admissions"},
    ),
    "d_icd_procedures": RelBenchTableSpec("d_icd_procedures", "icd_code_version", None),
    "procedures_icd": RelBenchTableSpec(
        "procedures_icd",
        None,
        "procedure_time",
        {
            "subject_id": "patients",
            "hadm_id": "admissions",
            "icd_code_version": "d_icd_procedures",
        },
    ),
    "drgcodes": RelBenchTableSpec(
        "drgcodes",
        None,
        "drg_time",
        {"subject_id": "patients", "hadm_id": "admissions"},
    ),
    "omr": RelBenchTableSpec(
        "omr",
        None,
        "omr_time",
        {"subject_id": "patients"},
    ),
}

DEFAULT_RELBENCH_TABLES = tuple(RELBENCH_TABLES)
