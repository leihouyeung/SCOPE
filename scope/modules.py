"""Building blocks for SCOPE.

    EncoderMLP                  per-modality 3-layer MLP -> shared 256-d space
    ProjectorMLP                3-layer BN-MLP for InfoNCE alignment only
    NeighborCrossAttention      Delaunay-neighbour cross-attention with
                                a learned per-head distance bias
    GatedFusion                 per-cell scalar (bi) or softmax (tri) gate
                                + residual-plus-projection rule
    GATModule                   2-layer GAT for spatial context aggregation
    ReconstructionHead          decodes the fused embedding back to each
                                modality (information-preservation guard)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ----------------------------- Encoder + projector -----------------------------

class EncoderMLP(nn.Module):
    """3-layer MLP: input_dim -> 2d -> d -> d, with GELU + LN + dropout."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(x))


class ProjectorMLP(nn.Module):
    """3-layer BN-MLP for the InfoNCE projector branch only."""

    def __init__(self, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------- Cross-attention -----------------------------

class NeighborCrossAttention(nn.Module):
    """Multi-head cross-attention restricted to first-order Delaunay neighbours.

    Each head adds a learned distance bias b^(h)(||p_i - p_j||) to its
    dot-product score, encoding the physical prior that proximate neighbours
    are typically more informative than distant ones while still allowing
    heads to specialise to short- and long-range contexts.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 16,
                 dropout: float = 0.1, use_distance_bias: bool = True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_distance_bias = use_distance_bias
        if use_distance_bias:
            self.dist_bias = nn.Sequential(
                nn.Linear(1, hidden_dim // 4), nn.ReLU(),
                nn.Linear(hidden_dim // 4, num_heads),
            )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                edge_index: torch.Tensor, edge_dist: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Args:
            query, key, value: (N, d) modality features (key == value here).
            edge_index: (2, E) directed edges; row = source, col = target.
            edge_dist: (E, 1) Euclidean distance per edge.
        """
        N, d = query.shape
        src, tgt = edge_index[0], edge_index[1]
        # Project: target attends to source.
        q = self.W_Q(query[tgt]).view(-1, self.num_heads, self.head_dim)
        k = self.W_K(key[src]).view(-1, self.num_heads, self.head_dim)
        v = self.W_V(value[src]).view(-1, self.num_heads, self.head_dim)

        # Per-edge per-head score.
        scores = (q * k).sum(-1) * self.scale                                    # (E, H)
        if self.use_distance_bias and edge_dist is not None:
            scores = scores + self.dist_bias(edge_dist)                           # (E, H)

        # Segment-softmax over each target's incoming edges, per head.
        max_score = torch.full((N, self.num_heads), -1e30, device=scores.device, dtype=scores.dtype)
        max_score.scatter_reduce_(0, tgt.unsqueeze(-1).expand_as(scores), scores, reduce="amax")
        exp = (scores - max_score[tgt]).exp()
        denom = torch.zeros(N, self.num_heads, device=exp.device, dtype=exp.dtype)
        denom.scatter_add_(0, tgt.unsqueeze(-1).expand_as(exp), exp)
        attn = exp / (denom[tgt] + 1e-12)                                         # (E, H)

        # Weighted sum.
        out = (attn.unsqueeze(-1) * v).reshape(-1, d)                             # (E, d)
        agg = torch.zeros(N, d, device=out.device, dtype=out.dtype)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(out), out)

        return self.norm(query + self.dropout(self.W_O(agg)))


# ----------------------------- Adaptive gating fusion -----------------------------

class GatedFusion(nn.Module):
    """Per-cell adaptive fusion.

    For 2 modalities: scalar sigmoid gate alpha_i; fused = MLP([a,b]) +
        W_proj(alpha*a + (1-alpha)*b). The first term lets a two-layer MLP
        learn non-linear cross-modal interactions; the second is a gated
        linear residual that keeps alpha directly interpretable.

    For >=3 modalities: softmax gate produces a simplex weight; fused is the
        convex combination of per-modality context vectors.
    """

    def __init__(self, hidden_dim: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        if num_modalities == 2:
            gate_input = hidden_dim * 2
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.fusion_net = nn.Sequential(
                nn.Linear(gate_input, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            )
            self.weighted_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            gate_input = hidden_dim * num_modalities
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_modalities),
            )

    def forward(self, context_vectors: list[torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            context_vectors: list of (N, d) per-modality context tensors.

        Returns:
            fused: (N, d) fused representation.
            alpha: (N, 1) scalar gate for bi-modal, (N, M) simplex weight otherwise.
        """
        if self.num_modalities == 2:
            a, b = context_vectors
            combined = torch.cat([a, b], dim=1)
            alpha = torch.sigmoid(self.gate_net(combined))                          # (N, 1)
            weighted = alpha * a + (1 - alpha) * b
            fused = self.fusion_net(combined) + self.weighted_proj(weighted)
            return fused, alpha
        else:
            combined = torch.cat(context_vectors, dim=1)
            alpha = torch.softmax(self.gate_net(combined), dim=1)                   # (N, M)
            fused = sum(alpha[:, k:k + 1] * v for k, v in enumerate(context_vectors))
            return fused, alpha


# ----------------------------- Spatial GAT -----------------------------

class GATModule(nn.Module):
    """Two-layer graph attention with initial-residual connection.

    Cross-attention reconciles multiple modalities of the same cell; this GAT
    propagates context across neighbouring cells to encode the local niche.
    Two layers and a residual connection are used to increase contextual
    awareness while limiting over-smoothing on dense biological graphs.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 num_heads: int = 16, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for layer in range(num_layers):
            heads = num_heads if layer < num_layers - 1 else 1
            concat = layer < num_layers - 1
            out_dim = hidden_dim if concat else hidden_dim
            layers.append(GATConv(in_dim, out_dim, heads=heads, concat=concat, dropout=dropout))
            in_dim = hidden_dim * (heads if concat else 1)
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
                                    for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            if i < len(self.layers) - 1:
                h = F.gelu(h)
                h = self.dropout(h)
        residual = x if self.residual_proj is None else self.residual_proj(x)
        return h + residual


# ----------------------------- Reconstruction head -----------------------------

class ReconstructionHead(nn.Module):
    """Two-layer MLP that decodes the fused embedding back to a modality.

    Forces the latent state to remain information-preserving rather than
    collapsing onto a purely task-driven bottleneck.
    """

    def __init__(self, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

