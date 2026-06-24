"""Training loop and inference for SCOPE.

Implements the three-part objective L_total = w_a L_align + w_r L_recon
+ w_c L_cluster, with a cross-batch memory queue of InfoNCE negatives and
Hilbert-ordered mini-batches that occupy compact tissue regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm   # ipywidget bar in notebooks, terminal bar in CLI

from .graph import (
    build_adjacency, build_delaunay_graph, expand_core_halo, hilbert_sort_indices,
)
from .losses import (
    cosine_reconstruction_loss, dec_cluster_loss, initialize_cluster_centers,
    pairwise_infonce_loss,
)
from .memory import CrossBatchMemory
from .model import SCOPE


# --------------------------- Hard-coded internals --------------------------
# These values follow the paper's tri-modal setting and have been validated
# across the reported datasets. They are kept here -- not on the CLI -- to
# stop reviewers reinventing them by accident.
_TEMPERATURE = 0.5            # InfoNCE temperature
_WEIGHT_DECAY = 0.0           # Adam weight decay
_PATIENCE = 30                # Early-stopping patience (epochs)
_ACCUMULATION_STEPS = 1       # Gradient accumulation
_MIXED_PRECISION = True       # torch.amp autocast
_HALO_HOPS = 2                # BFS hops added around each Hilbert batch
_SEED = 42                    # Random seed
_NAN_TOLERANCE = 5            # Abort after this many consecutive NaN steps


@dataclass
class TrainConfig:
    """User-tunable training hyperparameters."""
    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 1e-3
    w_align: float = 1.0
    w_recon: float = 10.0
    w_cluster: float = 5.0
    num_clusters: int = 12
    xbm_size: int = 65536
    device: str = "cuda"


# ----------------------------- Trainer -----------------------------

class SCOPETrainer:
    """End-to-end SCOPE training and inference."""

    def __init__(self, model: SCOPE, config: TrainConfig):
        self.model = model.to(config.device)
        self.config = config
        self.cluster_centers: Optional[torch.Tensor] = None
        torch.manual_seed(_SEED)
        np.random.seed(_SEED)

    # ------------------------------------------------------------------
    def fit(self, inputs: dict[str, torch.Tensor], coords: np.ndarray,
            cell_ids: Optional[np.ndarray] = None) -> None:
        """Train the SCOPE model.

        Args:
            inputs: dict {modality_name: (N, D_m) tensor}.
            coords: (N, 2) cell coordinates in micrometres.
            cell_ids: optional (N,) unique identifiers used to mask same-cell
                positives in the memory queue. Defaults to ``np.arange(N)``.
        """
        cfg = self.config
        names = self.model.modality_names

        # --- Spatial graph + Hilbert ordering.
        edge_index, _ = build_delaunay_graph(coords)
        adjacency = build_adjacency(edge_index, num_nodes=coords.shape[0])
        hilbert_order = hilbert_sort_indices(coords)

        # --- Memory queue.
        proj_dim = next(iter(self.model.projectors.values())).net[-1].out_features
        memory = CrossBatchMemory({n: proj_dim for n in names}, cfg.xbm_size, cfg.device)

        # --- Send to device.
        device = cfg.device
        inputs = {n: v.to(device).float() for n, v in inputs.items()}
        coords_t = torch.from_numpy(coords).to(device).float()
        edge_index = edge_index.to(device)
        if cell_ids is None:
            cell_ids = np.arange(coords.shape[0], dtype=np.int64)
        cell_ids_t = torch.from_numpy(cell_ids).to(device).long()

        # --- Optimiser + schedule.
        optim_ = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=_WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optim_, T_max=cfg.epochs)

        # Use the new torch.amp API.
        use_cuda_amp = _MIXED_PRECISION and torch.cuda.is_available() and "cuda" in str(device)
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

        # --- Mini-batches.
        batches = [hilbert_order[i:i + cfg.batch_size]
                   for i in range(0, len(hilbert_order), cfg.batch_size)]
        if not batches:
            raise RuntimeError(
                "No mini-batches were created. Check that batch_size > 0 and "
                "that the input has at least one cell."
            )

        best_loss, best_state, patience = float("inf"), None, 0
        nan_strikes = 0
        pbar = tqdm(range(cfg.epochs), desc="Training", dynamic_ncols=True)
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for step, core in enumerate(batches):
                core_t = torch.from_numpy(core).to(device).long()
                if _HALO_HOPS > 0:
                    expanded = expand_core_halo(adjacency, core, _HALO_HOPS)
                else:
                    expanded = np.unique(core)
                expanded_t = torch.from_numpy(expanded).to(device).long()

                sub_inputs, sub_edges, sub_coords, core_local = self._subgraph(
                    inputs, edge_index, coords_t, expanded_t, core_t,
                )

                with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                    z, extras = self.model(sub_inputs, sub_edges, sub_coords,
                                           return_extras=True)

                    # --- Loss components on the core sub-batch only.
                    proj = {n: extras["proj"][n][core_local] for n in names}
                    q_ids, q_proj = memory.get()
                    batch_ids = cell_ids_t[core_t]
                    l_align = pairwise_infonce_loss(
                        proj, temperature=_TEMPERATURE,
                        queue_ids=q_ids, queue_projections=q_proj,
                        batch_ids=batch_ids,
                    )

                    recon_core = {n: extras["recon"][n][core_local] for n in names}
                    targets_core = {n: sub_inputs[n][core_local] for n in names}
                    input_norms = {n: self.model.encoders[n].input_norm for n in names}
                    l_recon = cosine_reconstruction_loss(recon_core, targets_core, input_norms)

                    if self.cluster_centers is None and epoch >= 1:
                        # K-means crashes when n_samples < n_clusters; clamp.
                        z_core = z[core_local].detach().float()
                        k_eff = max(2, min(cfg.num_clusters, z_core.shape[0]))
                        if k_eff < cfg.num_clusters:
                            print(f"[warn] num_clusters={cfg.num_clusters} > batch "
                                  f"size {z_core.shape[0]}; using k={k_eff}.")
                        self.cluster_centers = initialize_cluster_centers(
                            z_core, k_eff, seed=_SEED,
                        )
                        self.cluster_centers = nn.Parameter(self.cluster_centers)
                        optim_.add_param_group({"params": [self.cluster_centers]})
                    l_cluster = (dec_cluster_loss(z[core_local], self.cluster_centers)
                                 if self.cluster_centers is not None
                                 else torch.zeros((), device=device))

                    loss = (cfg.w_align * l_align +
                            cfg.w_recon * l_recon +
                            cfg.w_cluster * l_cluster) / _ACCUMULATION_STEPS

                # NaN / Inf guard.
                if not torch.isfinite(loss):
                    nan_strikes += 1
                    tqdm.write(
                        f"[warn] non-finite loss at epoch {epoch} step {step} "
                        f"(align={l_align.item():.3g}, recon={l_recon.item():.3g}, "
                        f"cluster={l_cluster.item():.3g}); skipping step."
                    )
                    optim_.zero_grad(set_to_none=True)
                    if nan_strikes >= _NAN_TOLERANCE:
                        raise RuntimeError(
                            f"Training aborted after {_NAN_TOLERANCE} consecutive "
                            "non-finite losses. Try a smaller learning rate."
                        )
                    continue
                nan_strikes = 0

                scaler.scale(loss).backward()
                if (step + 1) % _ACCUMULATION_STEPS == 0:
                    scaler.step(optim_)
                    scaler.update()
                    optim_.zero_grad(set_to_none=True)

                memory.enqueue(
                    ids=batch_ids,
                    projections={n: proj[n].detach() for n in names},
                )

                epoch_loss += float(loss.item()) * _ACCUMULATION_STEPS

            scheduler.step()
            epoch_loss /= max(1, len(batches))
            pbar.set_postfix(loss=f"{epoch_loss:.3f}")

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience > _PATIENCE:
                    tqdm.write(f"Early stopping at epoch {epoch}.")
                    break
        pbar.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, inputs: dict[str, torch.Tensor], coords: np.ndarray) -> np.ndarray:
        """Run inference and return the (N, HIDDEN_DIM) cell embedding.

        Uses the same Hilbert mini-batching + 2-hop halo subgraph extraction
        as ``fit`` so that whole-slide inputs (10^5-10^6 cells) fit on a
        single GPU. The output is gathered core-cell-by-core-cell into a
        full (N, HIDDEN_DIM) tensor, matching the order of ``coords``.
        """
        cfg = self.config
        self.model.eval()
        device = cfg.device

        edge_index, _ = build_delaunay_graph(coords)
        edge_index = edge_index.to(device)
        adjacency = build_adjacency(edge_index.cpu(), num_nodes=coords.shape[0])
        hilbert_order = hilbert_sort_indices(coords)

        coords_t = torch.from_numpy(coords).to(device).float()
        inputs = {n: v.to(device).float() for n, v in inputs.items()}

        batches = [hilbert_order[i:i + cfg.batch_size]
                   for i in range(0, len(hilbert_order), cfg.batch_size)]

        # Pre-allocate output. HIDDEN_DIM is whatever the encoder emits;
        # run a single tiny forward to find it without hard-coding.
        first_core = torch.from_numpy(batches[0][:1]).to(device).long()
        sub_in0, sub_e0, sub_c0, core_l0 = self._subgraph(
            inputs, edge_index, coords_t, first_core, first_core,
        )
        z0 = self.model(sub_in0, sub_e0, sub_c0, return_extras=False)
        hidden_dim = z0.shape[-1]
        out = torch.empty((coords.shape[0], hidden_dim),
                          dtype=torch.float32, device="cpu")

        for core in tqdm(batches, desc="Inference"):
            core_t = torch.from_numpy(core).to(device).long()
            if _HALO_HOPS > 0:
                expanded = expand_core_halo(adjacency, core, _HALO_HOPS)
            else:
                expanded = np.unique(core)
            expanded_t = torch.from_numpy(expanded).to(device).long()

            sub_inputs, sub_edges, sub_coords, core_local = self._subgraph(
                inputs, edge_index, coords_t, expanded_t, core_t,
            )
            z_sub = self.model(sub_inputs, sub_edges, sub_coords,
                               return_extras=False)
            out[core_t.cpu()] = z_sub[core_local].float().cpu()

        return out.numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def _subgraph(inputs, edge_index, coords, node_ids, core_ids):
        """Extract a local subgraph containing `node_ids` and relabel edges."""
        device = edge_index.device
        node_ids = node_ids.to(device)
        relabel = torch.full((coords.shape[0],), -1, dtype=torch.long, device=device)
        relabel[node_ids] = torch.arange(node_ids.numel(), device=device)
        keep = (relabel[edge_index[0]] >= 0) & (relabel[edge_index[1]] >= 0)
        sub_edges = torch.stack([relabel[edge_index[0][keep]],
                                 relabel[edge_index[1][keep]]], dim=0)
        sub_inputs = {n: x[node_ids] for n, x in inputs.items()}
        sub_coords = coords[node_ids]
        core_local = relabel[core_ids.to(device)]
        return sub_inputs, sub_edges, sub_coords, core_local
