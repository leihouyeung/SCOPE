"""SCOPE: Single-Cell multimOdal sPatial intEgration.

A unified bi-/tri-modal cell-state representation framework. The same
backbone supports any number of input modalities; the cross-attention
topology and gating module adapt automatically.
"""
from __future__ import annotations

from itertools import permutations
from typing import Optional, Union

import torch
import torch.nn as nn

from .modalities import MODALITY_REGISTRY
from .modules import (
    EncoderMLP, GatedFusion, GATModule, NeighborCrossAttention,
    ProjectorMLP, ReconstructionHead,
)

# ----------------------------- Constants -----------------------------

HIDDEN_DIM = 256
PROJ_DIM = 128
NUM_HEADS = 16
NUM_GAT_LAYERS = 2
DROPOUT = 0.1


class SCOPE(nn.Module):
    """Unified spatial multi-modal integrator.

    Architecture:
        1. Per-modality LayerNorm + 3-layer MLP encoder -> shared 256-d space.
        2. Spatially-aware neighbour cross-attention between all modality pairs.
        3. Per-cell adaptive gating fusion (sigmoid for 2, softmax for >=3).
        4. Spatial GAT aggregation over the Delaunay graph.
        5. Per-modality reconstruction head (information-preservation guard).
        6. Per-modality projector branch for InfoNCE alignment (training only).

    Two construction modes are supported:

        # 1. List of registered modality names; dims taken from MODALITY_REGISTRY.
        SCOPE(modalities=["histology", "transcriptomics", "proteomics"])

        # 2. Explicit dims (overrides the registry; required for any modality
        #    whose dim is variable, e.g., proteomics).
        SCOPE(modality_dims={"histology": 768,
                             "transcriptomics": 64,
                             "proteomics": 42})

    When both `modalities` and `modality_dims` are provided, names must
    agree; dims from `modality_dims` win.
    """

    def __init__(
        self,
        modalities: Optional[list[str]] = None,
        modality_dims: Optional[dict[str, int]] = None,
    ):
        super().__init__()
        modality_dims = self._resolve_dims(modalities, modality_dims)

        if len(modality_dims) < 2:
            raise ValueError(
                f"SCOPE requires at least two modalities for cross-attention; "
                f"got {len(modality_dims)} ({list(modality_dims)})."
            )
        if HIDDEN_DIM % NUM_HEADS != 0:
            raise ValueError(
                f"HIDDEN_DIM ({HIDDEN_DIM}) must be divisible by NUM_HEADS ({NUM_HEADS})."
            )

        self.modality_names = tuple(modality_dims.keys())
        self.modality_dims = dict(modality_dims)
        self.hidden_dim = HIDDEN_DIM

        # Per-modality encoder + reconstruction head.
        self.encoders = nn.ModuleDict({
            name: EncoderMLP(dim, HIDDEN_DIM, DROPOUT)
            for name, dim in modality_dims.items()
        })
        self.decoders = nn.ModuleDict({
            name: ReconstructionHead(HIDDEN_DIM, dim, DROPOUT)
            for name, dim in modality_dims.items()
        })

        # Projector branch used by InfoNCE only (discarded at inference).
        self.projectors = nn.ModuleDict({
            name: ProjectorMLP(HIDDEN_DIM, PROJ_DIM) for name in self.modality_names
        })

        # Directed cross-attention paths for every ordered modality pair.
        # Modality `a` attends over its neighbours in modality `b`.
        self.cross_attn = nn.ModuleDict({
            f"{a}__{b}": NeighborCrossAttention(HIDDEN_DIM, NUM_HEADS, DROPOUT,
                                                use_distance_bias=True)
            for a, b in permutations(self.modality_names, 2)
        })

        # Adaptive gating fusion.
        self.fusion = GatedFusion(HIDDEN_DIM, num_modalities=len(self.modality_names),
                                  dropout=DROPOUT)

        # Spatial GAT over the Delaunay graph.
        self.gat = GATModule(
            input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS, num_layers=NUM_GAT_LAYERS, dropout=DROPOUT,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_dims(modalities: Optional[list[str]],
                      modality_dims: Optional[dict[str, int]]
                      ) -> dict[str, int]:
        """Build the final {name: dim} dict from the two construction modes."""
        if modalities is None and modality_dims is None:
            raise ValueError(
                "Provide either `modalities` (list of registered names) or "
                "`modality_dims` (dict)."
            )

        out: dict[str, int] = {}

        # Start from the modalities list, falling back to registry defaults.
        if modalities is not None:
            for name in modalities:
                if name not in MODALITY_REGISTRY:
                    raise ValueError(
                        f"Unknown modality '{name}'. Registered modalities: "
                        f"{sorted(MODALITY_REGISTRY)}."
                    )
                reg_dim = MODALITY_REGISTRY[name].default_dim
                if reg_dim is None and (modality_dims is None or name not in modality_dims):
                    raise ValueError(
                        f"Modality '{name}' has no default feature dimension "
                        f"(variable per dataset). Pass `modality_dims={{'{name}': D}}` "
                        f"explicitly, or load the data first and call "
                        f"`SCOPE.from_inputs(inputs)`."
                    )
                if reg_dim is not None:
                    out[name] = int(reg_dim)

        # Overrides / additions from `modality_dims`.
        if modality_dims is not None:
            for name, dim in modality_dims.items():
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError(
                        f"modality '{name}' has invalid dim {dim!r}; must be a positive int."
                    )
                out[name] = int(dim)

        if not out:
            raise ValueError("Resolved modality_dims is empty.")
        return out

    @classmethod
    def from_inputs(cls, inputs: dict[str, "torch.Tensor"]) -> "SCOPE":
        """Build a SCOPE model with feature dimensions inferred from `inputs`.

        Use this when you have the per-modality tensors loaded and want
        SCOPE to figure out every per-modality dimension automatically --
        including the variable-dimension modalities (e.g. proteomics, whose
        channel count differs between panels).
        """
        dims = {name: int(tensor.shape[-1]) for name, tensor in inputs.items()}
        return cls(modality_dims=dims)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        return_extras: bool = False,
    ):
        """Args:
            inputs: dict of (N, D_m) raw modality features.
            edge_index: (2, E) Delaunay edges (with self-loops).
            coords: (N, 2) cell coordinates; used for per-edge distance bias.
            return_extras: also return per-modality reconstructions, projector
                embeddings and the fusion gate alpha (for training and
                interpretability).

        Returns:
            z: (N, HIDDEN_DIM) final cell embedding.
            (extras): optional dict with keys 'recon', 'proj', 'alpha'.
        """
        names = self.modality_names

        # ---- Stage I: input LayerNorm + symmetric encoder MLP.
        H = {n: self.encoders[n](inputs[n]) for n in names}

        # ---- Stage II: pairwise spatial-aware neighbour cross-attention.
        row, col = edge_index[0], edge_index[1]
        edge_dist = (coords[row] - coords[col]).norm(dim=1, keepdim=True).to(H[names[0]].dtype)

        attended = {n: [] for n in names}
        for a, b in permutations(names, 2):
            attn_out = self.cross_attn[f"{a}__{b}"](H[a], H[b], H[b], edge_index, edge_dist)
            attended[a].append(attn_out)

        # ---- Per-modality context: self + incoming cross-attention outputs.
        if len(names) == 2:
            # In the bi-modal case the two directed cross-attention outputs
            # feed the scalar gate directly.
            context_vectors = [attended[names[0]][0], attended[names[1]][0]]
        else:
            context_vectors = [
                (H[n] + sum(attended[n])) / (1 + len(attended[n])) for n in names
            ]

        # ---- Stage III: adaptive gating fusion.
        fused, alpha = self.fusion(context_vectors)

        # ---- Stage IV: spatial GAT.
        z = self.gat(fused, edge_index)

        if not return_extras:
            return z

        # ---- Stage V+VI extras: reconstruction and projector outputs for training.
        extras = {
            "recon": {n: self.decoders[n](z) for n in names},
            "proj": {n: self.projectors[n](H[n]) for n in names},
            "alpha": alpha,
        }
        return z, extras
