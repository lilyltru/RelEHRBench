"""Graph construction helpers for the GNN reproduction."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

from relbench.base import Database
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time


ColStats = Dict[str, Dict[str, Dict[StatType, Any]]]

# adapted from https://github.com/stanford-star/relbench/blob/main/relbench/modeling/graph.py
def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    col_stats_dict: Optional[ColStats] = None,
) -> Tuple[HeteroData, ColStats]:
    """Build a pkey-fkey graph, optionally reusing training column statistics.

    This is the RelBench 2.1.1 graph builder with one compatibility extension:
    ``col_stats_dict`` is forwarded table-by-table to TorchFrame materialization.
    Passing the training statistics for validation and test graphs keeps their
    categorical vocabularies and all other fitted statistics aligned with train.
    """
    data = HeteroData()
    materialized_col_stats: ColStats = {}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        df = table.df
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name].copy()
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:
            col_to_stype = {"__const__": stype.numerical}
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )
        table_col_stats = None if col_stats_dict is None else col_stats_dict[table_name]
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path, col_stats=table_col_stats) # difference here

        data[table_name].tf = dataset.tensor_frame
        materialized_col_stats[table_name] = dataset.col_stats

        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()
    return data, materialized_col_stats
