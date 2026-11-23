"""Joint-split adapter for the StageNet and Transformer tasks.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pyhealth.tasks.base_task import BaseTask  # type: ignore[reportMissingImports]


LABEL_OBSERVATION_END_KEY = "label_observation_end"
FUTURE_LABEL_TASKS = {"readmission_30d", "high_risk_lab_24h"}
ADMISSION_LABEL_TASKS = {
    "digestive_subtype",
    "cardiovascular_subtype",
    "phenotyping",
}
JOINT_TASKS = FUTURE_LABEL_TASKS | ADMISSION_LABEL_TASKS


def _discharge_times(patient: Any) -> Dict[str, Any]:
    """Map normalized admission IDs to observed discharge timestamps."""
    discharge_times: Dict[str, Any] = {}
    for discharge in patient.get_events(event_type="admission_discharges"):
        hadm_id = discharge.attr_dict.get("hadm_id")
        if hadm_id is not None and discharge.timestamp is not None:
            discharge_times[str(hadm_id)] = discharge.timestamp
    return discharge_times


def _with_label_observation_end(
    sample: Dict[str, Any],
    label_observation_end: Any,
) -> Dict[str, Any]:
    """Copy a task sample and attach its complete-label observation end."""
    return {
        **sample,
        LABEL_OBSERVATION_END_KEY: label_observation_end.isoformat(),
    }


class JointTaskAdapter(BaseTask):
    """Add joint-split label-completeness metadata to an existing task."""

    def __init__(self, task: BaseTask, task_name: str) -> None:
        if task_name not in JOINT_TASKS:
            raise KeyError(
                f"Unknown task_name={task_name!r}. "
                f"Available joint tasks: {sorted(JOINT_TASKS)}"
            )

        self.task = task
        self.short_task_name = task_name
        self.task_name = f"{task.task_name}_joint_split_v1"
        self.input_schema = task.input_schema
        self.output_schema = task.output_schema

        if task_name == "readmission_30d":
            # The joint time filter can verify the final discharge's complete
            # 30-day horizon, so it need not be dropped during sample creation.
            task.include_last_discharge = True

    def _future_label_samples(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        horizon = self.task.horizon
        return [
            _with_label_observation_end(
                sample,
                datetime.fromisoformat(sample["timestamp"]) + horizon,
            )
            for sample in samples
        ]

    @staticmethod
    def _completed_admission_samples(
        patient: Any,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        discharge_times = _discharge_times(patient)
        completed_samples: List[Dict[str, Any]] = []
        for sample in samples:
            discharge_time = discharge_times.get(str(sample["visit_id"]))
            if discharge_time is not None:
                completed_samples.append(
                    _with_label_observation_end(sample, discharge_time)
                )
        return completed_samples

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = self.task(patient)
        if self.short_task_name in FUTURE_LABEL_TASKS:
            return self._future_label_samples(samples)
        return self._completed_admission_samples(patient, samples)


def make_joint_task(task: BaseTask, task_name: str) -> BaseTask:
    """Wrap a StageNet or Transformer task with joint-split semantics."""
    return JointTaskAdapter(task, task_name)
