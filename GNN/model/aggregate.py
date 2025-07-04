import torch
import torch.nn as nn

from model.pna_utils import *
from .moh import MoHAttention

class PNAAggregator(nn.Module):
    """PNA Operator for aggregating"""
    def __init__(self, inp_dim, out_dim) -> None:
        super(PNAAggregator, self).__init__()

        self.aggregators = [aggregate_mean, aggregate_min, aggregate_max, aggregate_std]
        self.scalers = [scale_identity] # , scale_amplification, scale_attenuation]
        self.linear = nn.Linear((len(self.aggregators) * len(self.scalers) + 1) * inp_dim, out_dim)
        att_embed_dim = 160
        att_num_heads = 8
        
        self.attn_layer = MoHAttention(
            dim=att_embed_dim,
            num_heads=att_num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.1,
            proj_drop=0.1,
            shared_head=2,
            routed_head=2,
            head_dim=16
        )

    def forward(self, nodes):
        h = nodes.mailbox["h"]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=h.shape[-2], log_degree=self.avg_d) for scale in self.scalers], dim=1)
        h = torch.cat([h, nodes.data["node_feat"]], dim=1)
        # h = self.attn_layer(h.unsqueeze(1)).squeeze(1)
        return {"node_feat": self.linear(h)}


class MinAggregator(nn.Module):
    """Min Operator for aggregating"""
    def __init__(self, inp_dim, out_dim) -> None:
        super(MinAggregator, self).__init__()
        self.linear = nn.Linear(2 * inp_dim, out_dim)

    def forward(self, nodes):
        h = nodes.mailbox["h"]
        h = torch.min(h, dim=1)[0]
        h = torch.cat([h, nodes.data["node_feat"]], dim=1)
        return {"node_feat": self.linear(h)}


class MaxAggregator(nn.Module):
    """Max Operator for aggregating"""
    def __init__(self, inp_dim, out_dim) -> None:
        super(MaxAggregator, self).__init__()
        self.linear = nn.Linear(2 * inp_dim, out_dim)

    def forward(self, nodes):
        h = nodes.mailbox["h"]
        h = torch.max(h, dim=1)[0]
        h = torch.cat([h, nodes.data["node_feat"]], dim=1)
        return {"node_feat": self.linear(h)}


class MeanAggregator(nn.Module):
    """Mean Operator for aggregating"""
    def __init__(self, inp_dim, out_dim) -> None:
        super(MeanAggregator, self).__init__()
        self.linear = nn.Linear(2 * inp_dim, out_dim)

    def forward(self, nodes):
        h = nodes.mailbox["h"]
        h = torch.mean(h, dim=1)
        h = torch.cat([h, nodes.data["node_feat"]], dim=1)
        return {"node_feat": self.linear(h)}