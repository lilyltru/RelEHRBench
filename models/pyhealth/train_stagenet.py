from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.models import StageNet

from ..paths import model_cache_dir, new_run_dir
from .dataset import (
    CONFIG_PATH,
    TABLE_ROOT,
    get_split_root,
    MimicHospPyHealthDataset,
)
from .common import (
    MULTILABEL_METRIC_THRESHOLD,
    TASK_METRICS,
    TASK_MONITORS,
    MultilabelThresholdTrainer,
    seed_everything,
)
from .splits import (
    SPLIT_PROTOCOLS,
    get_tables_for_split_protocol,
    make_split_datasets,
)
from .joint_task import make_joint_task
from .tasks import get_task


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train StageNet on PyHealth MIMIC-Hosp tasks."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_MONITORS),
        default="readmission_30d",
        help="Task short name to train.",
    )
    parser.add_argument(
        "--split-protocol",
        choices=SPLIT_PROTOCOLS,
        default="patient",
        help="Use the patient split or the joint patient-and-time split.",
    )
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--padding", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    seed_everything(args.seed, deterministic=args.deterministic)
    table_root = TABLE_ROOT
    split_root = get_split_root(args.split_id)
    model_cache = (
        model_cache_dir(
            model="stagenet",
            protocol=args.split_protocol,
            split_id=args.split_id,
            task=args.task,
        )
        / f"padding_{args.padding}"
    )
    output_path, run_id = new_run_dir(
        model="stagenet",
        protocol=args.split_protocol,
        split_id=args.split_id,
        task=args.task,
        seed=args.seed,
    )
    print("Output path:", output_path)

    tables = get_tables_for_split_protocol(args.task, args.split_protocol)
    task = get_task(args.task, padding=args.padding)
    if args.split_protocol == "joint":
        task = make_joint_task(task, args.task)
    label_key = next(iter(task.output_schema))

    base_dataset = MimicHospPyHealthDataset(
        root=table_root,
        tables=tables,
        config_path=CONFIG_PATH,
        cache_dir=model_cache,
        dev=False,
        num_workers=args.num_workers,
    )
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    print("Input schema:", sample_dataset.input_schema)
    if len(sample_dataset) == 0:
        raise RuntimeError("set_task produced no samples")

    split_datasets = make_split_datasets(
        sample_dataset,
        split_protocol=args.split_protocol,
        split_root=split_root,
        table_root=table_root,
    )
    print(
        "Split sizes:",
        {split: len(dataset) for split, dataset in split_datasets.items()},
    )

    train_loader = get_dataloader(
        split_datasets["train"],
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    val_loader = get_dataloader(
        split_datasets["val"],
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    test_loader = get_dataloader(
        split_datasets["test"],
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=args.embedding_dim,
        chunk_size=args.chunk_size,
        levels=args.levels,
        dropout=args.dropout,
    )

    batch = next(iter(train_loader))
    with torch.no_grad():
        model(**batch)

    trainer = MultilabelThresholdTrainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        metrics=TASK_METRICS[args.task],
        output_path=str(output_path),
        multilabel_threshold=MULTILABEL_METRIC_THRESHOLD,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor=TASK_MONITORS[args.task],
        optimizer_params={"lr": args.lr},
    )

    test_results = trainer.evaluate(test_loader)
    print("Test results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value}")

    return test_results


if __name__ == "__main__":
    main()
