"""Cross-batch memory queue for InfoNCE alignment.

A Hilbert-ordered batch from one compact tissue region contains few hard
negatives, which weakens contrastive learning. ``CrossBatchMemory`` is a
modality-generic FIFO queue of recent projector embeddings whose entries
are concatenated as additional negatives at each InfoNCE step.

It exposes:

    enqueue(ids, projections_dict)   add a batch of projector embeddings
    get()                            return (ids, {modality_name: tensor})
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class CrossBatchMemory:
    """Modality-generic FIFO of projector embeddings."""

    def __init__(self, modality_dim: dict[str, int], size: int,
                 device: str | torch.device = "cpu"):
        self.modality_names = tuple(modality_dim.keys())
        self.size = int(size)
        self.device = torch.device(device)
        self.ptr = 0
        self.is_full = False
        if self.size <= 0:
            self.ids = None
            self.features: dict[str, "torch.Tensor | None"] = {n: None for n in self.modality_names}
            return
        self.ids = torch.full((self.size,), -1, dtype=torch.long, device=self.device)
        self.features = {
            n: torch.zeros((self.size, d), device=self.device, dtype=torch.float32)
            for n, d in modality_dim.items()
        }

    def __len__(self) -> int:
        return self.size if self.is_full else self.ptr

    @torch.no_grad()
    def get(self):
        n = len(self)
        if n <= 0:
            return None, None
        return self.ids[:n], {name: feat[:n] for name, feat in self.features.items()}

    @torch.no_grad()
    def enqueue(self, ids: torch.Tensor, projections: dict[str, torch.Tensor]) -> None:
        if self.size <= 0 or ids.numel() == 0:
            return
        ids = ids.detach().to(self.device, dtype=torch.long).view(-1)
        feats = {n: F.normalize(p.detach().to(self.device).float(), dim=1)
                 for n, p in projections.items()}
        n = ids.numel()
        if n >= self.size:
            ids = ids[-self.size:]
            feats = {k: v[-self.size:] for k, v in feats.items()}
            self.ids[:] = ids
            for k in self.modality_names:
                self.features[k][:] = feats[k]
            self.ptr, self.is_full = 0, True
            return
        end = self.ptr + n
        if end <= self.size:
            self.ids[self.ptr:end] = ids
            for k in self.modality_names:
                self.features[k][self.ptr:end] = feats[k]
        else:
            k_split = self.size - self.ptr
            self.ids[self.ptr:] = ids[:k_split]
            self.ids[:n - k_split] = ids[k_split:]
            for name in self.modality_names:
                self.features[name][self.ptr:] = feats[name][:k_split]
                self.features[name][:n - k_split] = feats[name][k_split:]
        self.ptr = (self.ptr + n) % self.size
        if self.ptr == 0:
            self.is_full = True
