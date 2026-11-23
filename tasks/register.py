"""Register the submission-local datasets and tasks with RelBench."""

import argparse
from pathlib import Path

from relbench.datasets import get_dataset, register_dataset
from relbench.tasks import register_task

from .dataset import (
    JOINT_DATASET_NAME,
    PATIENT_DATASET_NAME,
    MimicJointSplitDataset,
    MimicPatientSplitDataset,
)
from .definitions import (
    AdmissionCCSRMultilabelTask,
    AdmissionCardiovascularSubtypeMultilabelTask,
    AdmissionDigestiveSubtypeMultilabelTask,
    HighRiskLabAbnormalPatientSplitTask,
    PatientHospitalAdmission30DPatientSplitTask,
)


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE_NAME = "relehr-v2_2_30k"
SPLIT_PROTOCOLS = ("patient", "joint")
TASKS = {
    "patient-hosp-admit-30d": PatientHospitalAdmission30DPatientSplitTask,
    "patient-high-risk-lab-abnormal-24h": HighRiskLabAbnormalPatientSplitTask,
    "patient-phenotyping": AdmissionCCSRMultilabelTask,
    "patient-cardio-subtype": AdmissionCardiovascularSubtypeMultilabelTask,
    "patient-digestive-subtype": AdmissionDigestiveSubtypeMultilabelTask,
}


def register_mimic(
    split_id: str,
    split_protocol: str = "joint",
    split_seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> str:
    """Register one patient-based protocol and its five tasks."""

    if split_protocol == "patient":
        dataset_name = PATIENT_DATASET_NAME
        dataset_cls = MimicPatientSplitDataset
    elif split_protocol == "joint":
        dataset_name = JOINT_DATASET_NAME
        dataset_cls = MimicJointSplitDataset
    else:
        raise ValueError(
            f"Unknown split_protocol={split_protocol!r}; expected patient or joint"
        )

    dataset_cache = (
        SUBMISSION_ROOT / "cache" / "datasets" / "relbench" / DATASET_CACHE_NAME
    )
    register_dataset(
        dataset_name,
        dataset_cls,
        cache_dir=str(dataset_cache),
        split_id=split_id,
        split_seed=split_seed,
        split_ratios=split_ratios,
    )

    task_cache_root = SUBMISSION_ROOT / "cache" / "tasks" / dataset_name / split_id
    for task_name, task_cls in TASKS.items():
        register_task(
            dataset_name,
            task_name,
            task_cls,
            cache_dir=str(task_cache_root / task_name),
        )
    return dataset_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and cache the patient split used by all models."
    )
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    args = parser.parse_args()
    dataset_name = register_mimic(
        split_id=args.split_id,
        split_protocol="patient",
        split_seed=args.split_seed,
        split_ratios=tuple(args.split_ratios),
    )
    dataset = get_dataset(dataset_name, download=False)
    splits = dataset.get_patient_split()
    print({name: len(ids) for name, ids in splits.items()})
    print(f"Original subject IDs: {dataset.original_split_dir}")


if __name__ == "__main__":
    main()
