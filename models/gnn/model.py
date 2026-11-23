from typing import Any, Dict, List

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP as MLPHead
from torch_geometric.typing import NodeType

from relbench.modeling.nn import (
    HeteroGAT,
    HeteroGraphSAGE,
    HeteroTemporalEncoder,
)

import torch_frame
from torch_frame.nn.models import ResNet, FTTransformer
from torch_frame.nn.models import MLP as MLPEncoder


TORCH_FRAME_ENCODERS = {
    "resnet": ResNet,
    "mlp": MLPEncoder,
    "fttransformer": FTTransformer,
}


class ClampedTimestampEncoder(torch_frame.nn.TimestampEncoder):
    r"""Timestamp encoder that clamps negative year offsets to zero (for patient split)."""

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        feat = feat.to(torch.float32)
        # Clamp year offset to avoid negative values entering positional encoding.
        feat_year = feat[..., :1] - self.min_year.view(1, -1, 1)
        feat_year = torch.clamp(feat_year, min=0)
        feat_rest = feat[..., 1:] / self.max_values.view(1, 1, -1)
        x = torch.cat(
            [self.positional_encoding(feat_year), self.cyclic_encoding(feat_rest)],
            dim=2,
        )
        x_lin = torch.einsum("ijkl,jklm->ijm", x, self.weight)
        x = x_lin + self.bias
        return x

# adapted HeteroEncoder from https://github.com/stanford-star/relbench/blob/main/relbench/modeling/nn.py
class HeteroEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (ClampedTimestampEncoder, {}), # clamped
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model
        print(f"encoder model {torch_frame_model_cls}")

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


# adapted from https://github.com/stanford-star/relbench/blob/main/examples/model.py
class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        gnn: str = "sage",
        gat_heads: int = 4,
        gat_dropout: float = 0.0,
        encoder: str = "resnet",
    ):
        super().__init__()

        encoder = str(encoder).lower()
        try:
            torch_frame_model_cls = TORCH_FRAME_ENCODERS[encoder]
        except KeyError as error:
            choices = ", ".join(TORCH_FRAME_ENCODERS)
            raise ValueError(
                f"Unknown encoder={encoder!r}. Expected one of: {choices}."
            ) from error

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_cls=torch_frame_model_cls,
        )
        self.temporal_encoder = HeteroTemporalEncoder(  # sin PE
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        gnn = str(gnn).lower()
        if gnn == "sage":
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
            )
        elif gnn == "gat":
            self.gnn = HeteroGAT(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                heads=gat_heads,
                num_layers=num_layers,
                dropout=gat_dropout,
            )
        else:
            raise ValueError(f"Unknown gnn={gnn!r}. Expected 'sage' or 'gat'.")
        self.head = MLPHead(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.reset_parameters()

    def __repr__(self):
        return f"Model(\n" f"  gnn={self.gnn},\n" f"  head={self.head},\n" f")"

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return self.head(x_dict[entity_table][: seed_time.size(0)])
