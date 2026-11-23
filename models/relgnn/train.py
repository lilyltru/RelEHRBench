# adapted from https://github.com/stanford-star/relbench/blob/main/examples/gnn_entity.py to relgnn trainer
# train.py
#   → model.py::Model
#     → relgnn_nn.py::RelGNN # from relgnn repo
#       → relgnn_hetero_conv.py
#         → relgnn_conv.py
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tasks.register import SPLIT_PROTOCOLS, register_mimic

from sentence_transformers import SentenceTransformer
from torch.nn import BCEWithLogitsLoss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
# from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.graph import get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from .atomic_routes import get_atomic_routes
from ..gnn.graph_utils import make_pkey_fkey_graph
from ..paths import model_cache_dir, new_run_dir
from .model import Model

from typing import List, Optional
from torch import Tensor

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        # Some RelBench datasets can contain missing values in text columns.
        # SentenceTransformers tokenizers expect strings.
        cleaned: List[str] = []
        for s in sentences:
            if s is None:
                cleaned.append("")
            elif isinstance(s, str):
                cleaned.append(s)
            else:
                # Handle NaN/float/other scalars.
                try:
                    if isinstance(s, float) and s != s:  # NaN
                        cleaned.append("")
                    else:
                        cleaned.append(str(s))
                except Exception:
                    cleaned.append("")
        return self.model.encode(cleaned, convert_to_tensor=True)


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="patient-hosp-admit-30d")
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
parser.add_argument(
    "--stypes-file",
    type=Path,
    default=None,
    help=(
        "JSON defining the input columns and their stypes. Use stypes.json for "
        "all features, stypes_matched.json for the main experiment reproduction."
    ),
)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_model_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--num_heads", type=int, default=1)
parser.add_argument("--simplified_MP", action="store_true")
parser.add_argument(
    "--encoder",
    type=str,
    default="resnet",
    choices=["resnet", "mlp", "fttransformer"],
    help="Torch Frame encoder architecture for node features.",
)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--node_tables",
    nargs="+",
    default=[
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "prescriptions",
        "labevents",
    ],
    help=(
        "Input node tables to keep; pass 'all' to use every table. The task's "
        "entity table must be included."
    ),
)
args = parser.parse_args()
args.dataset = register_mimic(
    split_id=args.split_id,
    split_protocol=args.split_protocol,
    split_seed=args.split_seed,
    split_ratios=tuple(args.split_ratios),
)
cache_root = model_cache_dir(
    model="relgnn",
    protocol=args.split_protocol,
    split_id=args.split_id,
    task=args.task,
)
output_path, run_id = new_run_dir(
    model="relgnn",
    protocol=args.split_protocol,
    split_id=args.split_id,
    task=args.task,
    seed=args.seed,
)

KEEP_NODE_TYPES = None if args.node_tables == ["all"] else args.node_tables
print(
    "[node_ablation] requested node tables: "
    f"{KEEP_NODE_TYPES if KEEP_NODE_TYPES is not None else 'all'}"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=False)
task: EntityTask = get_task(args.dataset, args.task, download=False)

val_table = task.get_table("val", mask_input_cols=False)
test_eval_table = task.get_table("test", mask_input_cols=False)

if args.stypes_file is not None:
    stypes_bytes = args.stypes_file.read_bytes()
    stypes_cache_key = hashlib.sha256(stypes_bytes).hexdigest()[:12]
    col_to_stype_dict = json.loads(stypes_bytes)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
else:
    stypes_cache_key = "inferred"
    col_to_stype_dict = get_stype_proposal(dataset.get_db())

run_config = {
    **vars(args),
    "stypes_file": str(args.stypes_file) if args.stypes_file is not None else None,
    "stypes_cache_key": stypes_cache_key,
    "cache_dir": str(cache_root),
    "output_dir": str(output_path),
    "run_id": run_id,
}
with (output_path / "run_config.json").open("w", encoding="utf-8") as file:
    json.dump(run_config, file, indent=2)

print("Run ID:", run_id)
print("Cache path:", cache_root)
print("Output path:", output_path)

split_dbs = {
    split: dataset.get_split_db(split, upto_test_timestamp=False)
    for split in ("train", "val", "test")
}

text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device="cpu"),
    batch_size=4,
)

materialized_cache_root = (
    cache_root / f"stypes_{stypes_cache_key}" / "train_fitted_v1"
)

print("Encoding split train")
train_data, train_col_stats = make_pkey_fkey_graph(
    split_dbs["train"],
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=text_embedder_cfg,
    cache_dir=str(materialized_cache_root / "train" / "materialized"),
)
data_per_split = {"train": (train_data, train_col_stats)}
print(
    f"[{args.split_protocol}] train graph: "
    f"{ {nt: int(train_data[nt].num_nodes) for nt in train_data.node_types} }"
)

for split in ("val", "test"):
    print(f"Encoding split {split} with train-fitted column statistics")
    split_data, split_col_stats = make_pkey_fkey_graph(
        split_dbs[split],
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=str(materialized_cache_root / split / "materialized"),
        col_stats_dict=train_col_stats,
    )
    data_per_split[split] = (split_data, split_col_stats)
    print(
        f"[{args.split_protocol}] {split} graph: "
        f"{ {nt: int(split_data[nt].num_nodes) for nt in split_data.node_types} }"
    )

# Optional node-type ablation: keep only selected node types and induced edges.
if KEEP_NODE_TYPES is not None:
    keep_node_types = set(KEEP_NODE_TYPES)
    if task.entity_table not in keep_node_types:
        raise ValueError(
            f"task.entity_table '{task.entity_table}' must be included in KEEP_NODE_TYPES"
        )
    print(f"[node_ablation] keeping node types: {sorted(keep_node_types)}")
    for split in ("train", "val", "test"):
        split_data, split_col_stats = data_per_split[split]
        missing_node_types = sorted(keep_node_types.difference(split_data.node_types))
        if missing_node_types:
            raise ValueError(
                f"[node_ablation] split={split} missing node types: {missing_node_types}"
            )

        for edge_type in list(split_data.edge_types):
            src, _, dst = edge_type
            if src not in keep_node_types or dst not in keep_node_types:
                del split_data[edge_type]
        for node_type in list(split_data.node_types):
            if node_type not in keep_node_types:
                del split_data[node_type]

        split_col_stats = {
            node_type: stats
            for node_type, stats in split_col_stats.items()
            if node_type in keep_node_types
        }
        data_per_split[split] = (split_data, split_col_stats)
        print(
            f"[node_ablation] {split} graph: "
            f"{ {nt: int(split_data[nt].num_nodes) for nt in split_data.node_types} }"
        )

def print_used_features(data_per_split: Dict[str, tuple]) -> None:
    for split, (split_data, _) in data_per_split.items():
        print(f"[features] split={split}")
        for node_type in split_data.node_types:
            col_names_dict = split_data[node_type].tf.col_names_dict
            total_cols = sum(len(cols) for cols in col_names_dict.values())
            print(f"  {node_type}: {total_cols} feature columns")
            for stype_key, cols in col_names_dict.items():
                stype_name = getattr(stype_key, "value", str(stype_key))
                print(f"    {stype_name}: {list(cols)}")
print_used_features(data_per_split)

# Model and atomic routes are initialized from the train graph's structure.
data, _ = data_per_split["train"]
atomic_routes = get_atomic_routes(data.edge_types)
if not atomic_routes:
    raise ValueError(
        "RelGNN found no atomic routes. Expected graph relations prefixed with 'f2p'."
    )
print(f"RelGNN atomic routes: {len(atomic_routes)}")

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "average_precision"
    higher_is_better = True
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    # Each split has its own graph; NeighborLoader's time_attr="time" plus
    # input_time enforces "only use rows with node.time <= seed_time" per patient.
    split_data, _ = data_per_split[split]
    loader_dict[split] = NeighborLoader(
        split_data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


model = Model(
    data=data,
    col_stats_dict=data_per_split["train"][1],
    num_model_layers=args.num_model_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
    atomic_routes=atomic_routes,
    num_heads=args.num_heads,
    simplified_MP=args.simplified_MP,
    encoder=args.encoder,
).to(device)
print("=============== Architecture ===============")
print(model)
print("============================================ ")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
state_dict = None
ckpt_path = output_path / f"best_model_{run_id}.pt"
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, val_table)
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] <= best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())
        torch.save(state_dict, ckpt_path)
        print(f"  -> New best {tune_metric}={best_val_metric:.4f}")


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, val_table)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred, test_eval_table)
print(f"Best Test metrics: {test_metrics}")

metrics_report_path = output_path / f"metrics_report_{run_id}.json"
with metrics_report_path.open("w", encoding="utf-8") as fp:
    json.dump(
        {
            "config": run_config,
            "val": {k: float(v) for k, v in val_metrics.items()},
            "test": {k: float(v) for k, v in test_metrics.items()},
        },
        fp,
        indent=2,
    )
