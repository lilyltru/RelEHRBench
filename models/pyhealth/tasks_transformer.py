from __future__ import annotations

from typing import Any, Dict, List, Type

from pyhealth.tasks.base_task import BaseTask  # type: ignore[reportMissingImports]

from .tasks import (
    AdmissionCardiovascularSubtypeStageNetTask,
    AdmissionCCSRPhenotypingStageNetTask,
    AdmissionDigestiveSubtypeStageNetTask,
    HighRiskLabAbnormal24HStageNetTask,
    Readmission30DStageNetTask,
)


class _TransformerTaskMixin:
    """Adapt an existing StageNet task to PyHealth's Transformer processor.

    The StageNet tasks return ``codes=(visit_intervals, visit_codes)``. 
    PyHealth's Transformer consumes the same visit-wise token lists through its
    ``nested_sequence`` processor.
    """

    def __init__(self, padding: int = 5):
        super().__init__(padding=padding)
        self.input_schema = {"codes": "nested_sequence"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        stagenet_samples = super().__call__(patient)
        transformer_samples: List[Dict[str, Any]] = []

        for sample in stagenet_samples:
            stage_codes = sample["codes"]
            if not isinstance(stage_codes, tuple) or len(stage_codes) != 2:
                raise ValueError(
                    "Expected StageNet codes as (visit_intervals, visit_codes), "
                    f"got {type(stage_codes).__name__}"
                )

            _, visit_codes = stage_codes
            transformer_sample = dict(sample)
            transformer_sample["codes"] = visit_codes
            transformer_samples.append(transformer_sample)

        return transformer_samples


class Readmission30DTransformerTask(
    _TransformerTaskMixin,
    Readmission30DStageNetTask,
):
    task_name = "readmission_30d_transformer_v1"


class HighRiskLabAbnormal24HTransformerTask(
    _TransformerTaskMixin,
    HighRiskLabAbnormal24HStageNetTask,
):
    task_name = "high_risk_lab_24h_transformer_v1"


class AdmissionDigestiveSubtypeTransformerTask(
    _TransformerTaskMixin,
    AdmissionDigestiveSubtypeStageNetTask,
):
    task_name = "digestive_subtype_transformer_v1"


class AdmissionCardiovascularSubtypeTransformerTask(
    _TransformerTaskMixin,
    AdmissionCardiovascularSubtypeStageNetTask,
):
    task_name = "cardiovascular_subtype_transformer_v1"


class AdmissionCCSRPhenotypingTransformerTask(
    _TransformerTaskMixin,
    AdmissionCCSRPhenotypingStageNetTask,
):
    task_name = "phenotyping_transformer_v1"


TRANSFORMER_TASK_REGISTRY: dict[str, Type[BaseTask]] = {
    "readmission_30d": Readmission30DTransformerTask,
    "high_risk_lab_24h": HighRiskLabAbnormal24HTransformerTask,
    "digestive_subtype": AdmissionDigestiveSubtypeTransformerTask,
    "cardiovascular_subtype": AdmissionCardiovascularSubtypeTransformerTask,
    "phenotyping": AdmissionCCSRPhenotypingTransformerTask,
}


def get_transformer_task(task_name: str) -> BaseTask:
    """Instantiate a Transformer task while reusing StageNet token construction."""
    if task_name not in TRANSFORMER_TASK_REGISTRY:
        raise KeyError(
            f"Unknown task_name={task_name!r}. "
            f"Available tasks: {sorted(TRANSFORMER_TASK_REGISTRY)}"
        )
    return TRANSFORMER_TASK_REGISTRY[task_name]()
