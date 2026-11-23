"""Configuration shared by the StageNet and Transformer trainers."""

import os
import random

import numpy as np
import torch
from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.trainer import Trainer

TASK_METRICS = {
    "readmission_30d": ["pr_auc", "roc_auc", "accuracy", "f1"],
    "high_risk_lab_24h": ["pr_auc", "roc_auc", "accuracy", "f1"],
    "digestive_subtype": [
        "pr_auc_micro",
        "pr_auc_macro",
        "roc_auc_micro",
        "roc_auc_macro",
        "f1_micro",
        "f1_macro",
    ],
    "cardiovascular_subtype": [
        "pr_auc_micro",
        "pr_auc_macro",
        "roc_auc_micro",
        "roc_auc_macro",
        "f1_micro",
        "f1_macro",
    ],
    "phenotyping": [
        "pr_auc_micro",
        "pr_auc_macro",
        "roc_auc_micro",
        "roc_auc_macro",
        "f1_micro",
        "f1_macro",
    ],
}

TASK_MONITORS = {
    "readmission_30d": "pr_auc",
    "high_risk_lab_24h": "pr_auc",
    "digestive_subtype": "pr_auc_macro",
    "cardiovascular_subtype": "pr_auc_macro",
    "phenotyping": "pr_auc_macro",
}

MULTILABEL_METRIC_THRESHOLD = 0.5 # pyhealth default 0.3, set to 0.5 to be consistent with relbench


class MultilabelThresholdTrainer(Trainer):
    """PyHealth trainer with an explicit multilabel decision threshold."""

    def __init__(
        self,
        *args,
        multilabel_threshold: float = MULTILABEL_METRIC_THRESHOLD,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= multilabel_threshold <= 1.0:
            raise ValueError("multilabel_threshold must be between 0 and 1")
        self.multilabel_threshold = multilabel_threshold

    def evaluate(self, dataloader) -> dict[str, float]:
        if getattr(self.model, "mode", None) != "multilabel":
            return super().evaluate(dataloader)

        y_true, y_prob, loss_mean = self.inference(dataloader)
        scores = multilabel_metrics_fn(
            y_true,
            y_prob,
            metrics=self.metrics,
            threshold=self.multilabel_threshold,
        )
        scores["loss"] = loss_mean
        return scores


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
