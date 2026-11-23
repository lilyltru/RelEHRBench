# adapted from https://github.com/stanford-star/relbench/blob/main/examples/lightgbm_entity.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
import relbench
import torch
import torch_frame
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tasks.register import SPLIT_PROTOCOLS, register_mimic

from ..paths import model_cache_dir, new_run_dir
from .features_allowed import (
    ALLOWED_ATTRIBUTE_FEATURE_VERSION,
    build_allowed_attribute_features,
)

SPLITS = ("train", "val", "test")


def build_arg_parser(default_task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=default_task)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50_000,
        help="Subsample this many training rows before fitting. Use 0 for all rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


class AUPRCLightGBM(LightGBM):
    """Use average precision as the Optuna trial-selection objective."""

    @torch.no_grad()
    def compute_metric(self, target: torch.Tensor, pred: torch.Tensor) -> float:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(target.cpu(), pred.cpu()))


def _fit_and_evaluate(
    task: EntityTask,
    train_dataset: torch_frame.data.Dataset,
    tensor_frames: Mapping[str, Any],
    target_tables: Mapping[str, Any],
    num_trials: int,
) -> Dict[str, Dict[str, float]]:
    tune_metric = Metric.ROCAUC

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        model = AUPRCLightGBM(task_type=train_dataset.task_type, metric=tune_metric)
        model.tune(
            tf_train=tensor_frames["train"],
            tf_val=tensor_frames["val"],
            num_trials=num_trials,
        )

        predictions = {
            split: model.predict(tf_test=tensor_frames[split]).numpy()
            for split in SPLITS
        }
        metrics = {
            split: task.evaluate(
                predictions[split],
                target_tables[split],
            )
            for split in SPLITS
        }
        return metrics

    if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        y_train = tensor_frames["train"].y.values.to(torch.long)
        y_val = tensor_frames["val"].y.values.to(torch.long)
        pred_lists = {split: [] for split in SPLITS}

        for label_idx in tqdm(range(task.num_labels)):
            model = AUPRCLightGBM(
                task_type=torch_frame.TaskType.BINARY_CLASSIFICATION,
                metric=tune_metric,
            )
            tensor_frames["train"].y = y_train[:, label_idx]
            tensor_frames["val"].y = y_val[:, label_idx]
            model.tune(
                tf_train=tensor_frames["train"],
                tf_val=tensor_frames["val"],
                num_trials=num_trials,
            )
            for split in SPLITS:
                pred_lists[split].append(
                    model.predict(tf_test=tensor_frames[split]).numpy()
                )

        predictions = {
            split: np.stack(pred_lists[split]).transpose() for split in SPLITS
        }
        metrics = {
            split: task.evaluate(
                predictions[split],
                target_tables[split],
            )
            for split in SPLITS
        }
        return metrics

    raise ValueError(f"Unsupported task type: {task.task_type}")


def run_lightgbm_entity(
    args: argparse.Namespace,
    split_kind: str,
    feature_builder: Callable[[Any, EntityTask, Any], Any],
    feature_version: str,
    feature_label: str,
) -> Dict[str, Dict[str, float]]:
    print("relbench source:", Path(relbench.__file__).resolve())
    print("dataset:", args.dataset)
    print("task:", args.task)
    print("split_kind:", split_kind)

    if torch.cuda.is_available():
        torch.set_num_threads(1)
    seed_everything(args.seed)

    dataset: Dataset = get_dataset(args.dataset, download=False)
    task: EntityTask = get_task(args.dataset, args.task, download=False)
    if task.task_type not in (
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
    ):
        raise ValueError(f"Unsupported task type: {task.task_type}")
    tuning_objective = "average_precision"
    print("tuning objective:", tuning_objective)

    feature_tables = {
        "train": task.get_table("train", mask_input_cols=False),
        "val": task.get_table("val", mask_input_cols=False),
        "test": task.get_table("test", mask_input_cols=True),
    }
    target_tables = {
        split: task.get_table(split, mask_input_cols=False) for split in SPLITS
    }

    if args.sample_size > 0 and args.sample_size < len(feature_tables["train"]):
        sampled_idx = np.random.default_rng(args.seed).permutation(
            len(feature_tables["train"])
        )[: args.sample_size]
        sampled_df = feature_tables["train"].df.iloc[sampled_idx].reset_index(drop=True)
        feature_tables["train"].df = sampled_df
        target_tables["train"].df = sampled_df.copy()

    dfs = {
        split: feature_builder(
            dataset.get_split_db(split, upto_test_timestamp=False), task, table
        )
        for split, table in feature_tables.items()
    }

    feature_cols = [
        column
        for column in dfs["train"].columns
        if column != task.target_col and dfs["train"][column].notna().any()
    ]
    dropped_cols = set(dfs["train"]) - set(feature_cols) - {task.target_col}
    for split in SPLITS:
        keep = feature_cols + (
            [task.target_col] if task.target_col in dfs[split] else []
        )
        dfs[split] = dfs[split].reindex(columns=keep)

    col_to_stype = {column: torch_frame.numerical for column in feature_cols}
    if "gender" in col_to_stype:
        col_to_stype["gender"] = torch_frame.categorical
    print(f"{feature_label} features: {len(feature_cols)}")
    if dropped_cols:
        print(f"dropped all-missing features: {sorted(dropped_cols)}")

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        col_to_stype[task.target_col] = torch_frame.categorical
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        col_to_stype[task.target_col] = torch_frame.embedding
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")

    train_dataset = torch_frame.data.Dataset(
        df=dfs["train"],
        col_to_stype=col_to_stype,
        target_col=task.target_col,
    )
    materialized_path = (
        Path(args.cache_dir)
        / feature_version
        / f"seed_{args.seed}_sample_{args.sample_size}"
        / "materialized"
        / "node_train.pt"
    )
    materialized_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset = train_dataset.materialize(path=materialized_path)

    tensor_frames = {
        "train": train_dataset.tensor_frame,
        "val": train_dataset.convert_to_tensor_frame(dfs["val"]),
        "test": train_dataset.convert_to_tensor_frame(dfs["test"]),
    }
    metrics = _fit_and_evaluate(
        task=task,
        train_dataset=train_dataset,
        tensor_frames=tensor_frames,
        target_tables=target_tables,
        num_trials=args.num_trials,
    )
    metrics = {
        split: {name: float(value) for name, value in split_metrics.items()}
        for split, split_metrics in metrics.items()
    }

    for split in SPLITS:
        print(f"{split.title()}: {metrics[split]}")

    run_dir, output_run_id = new_run_dir(
        model="lightgbm",
        protocol=split_kind,
        split_id=args.split_id,
        task=args.task,
        seed=args.seed,
    )
    print("Output path:", run_dir)
    result_path = run_dir / f"metrics_report_{output_run_id}.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "task": args.task,
                "split_kind": split_kind,
                "split_id": args.split_id,
                "cache_dir": args.cache_dir,
                "output_dir": str(run_dir),
                "run_id": output_run_id,
                "seed": args.seed,
                "sample_size": args.sample_size,
                "num_trials": args.num_trials,
                "feature_version": feature_version,
                "tuning_objective": tuning_objective,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    return metrics


def main() -> None:
    parser = build_arg_parser(default_task="patient-hosp-admit-30d")
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--split-protocol", choices=SPLIT_PROTOCOLS, default="joint")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    args = parser.parse_args()
    args.dataset = register_mimic(
        split_id=args.split_id,
        split_protocol=args.split_protocol,
        split_seed=args.split_seed,
        split_ratios=tuple(args.split_ratios),
    )
    args.cache_dir = str(
        model_cache_dir(
            model="lightgbm",
            protocol=args.split_protocol,
            split_id=args.split_id,
            task=args.task,
        )
    )

    args.model = "lightgbm"

    # Joint datasets apply their cumulative time window inside get_split_db().
    run_lightgbm_entity(
        args,
        split_kind=args.split_protocol,
        feature_builder=build_allowed_attribute_features,
        feature_version=ALLOWED_ATTRIBUTE_FEATURE_VERSION,
        feature_label="allowed-attribute point-in-time",
    )


if __name__ == "__main__":
    main()
