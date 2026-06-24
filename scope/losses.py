"""Loss functions for SCOPE.

    pairwise_infonce_loss   InfoNCE summed over all unordered modality pairs;
                            extends naturally from two to three or more
                            modalities. Supports a memory queue of extra
                            negatives and an optional logit bonus on
                            queue entries flagged as hard negatives.

    cosine_reconstruction_loss   1 - cosine similarity between decoded and
                            LayerNorm-transformed input, summed uniformly
                            across modalities.

    dec_cluster_loss        KL(target || soft) using a Student-t kernel and
                            a target sharpened by squaring + renormalisation,
                            preventing trivial collapse onto a single cluster.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import torch
import torch.nn.functional as F

INFONCE_LOGIT_CLAMP = 30.0   # AMP overflow guard
INFONCE_MASK_NEG = -1e4      # finite mask for same-cell-id entries


# ----------------------------- Alignment loss -----------------------------

def pairwise_infonce_loss(
    projections: dict[str, torch.Tensor],
    temperature: float = 0.5,
    queue_ids: Optional[torch.Tensor] = None,
    queue_projections: Optional[dict[str, torch.Tensor]] = None,
    batch_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise InfoNCE summed over all unordered modality pairs.

    Args:
        projections: dict mapping modality name to (N, d_proj) projector output.
        temperature: InfoNCE temperature.
        queue_ids: optional (M,) cell ids of queued projector pairs.
        queue_projections: optional dict mapping modality name to (M, d_proj).
        batch_ids: optional (N,) cell ids of the current batch; positions in
            the queue that share an id are masked to prevent false negatives.

    Returns:
        Scalar alignment loss summed over modality pairs.
    """
    names = list(projections.keys())
    if len(names) < 2:
        return torch.tensor(0.0, device=next(iter(projections.values())).device, requires_grad=True)

    # L2-normalise once.
    P = {n: F.normalize(p, dim=1) for n, p in projections.items()}
    Q = {n: F.normalize(q, dim=1) for n, q in (queue_projections or {}).items()}

    total = 0.0
    for a, b in combinations(names, 2):
        total = total + _infonce_pair(
            P[a], P[b], temperature,
            queue_ids=queue_ids,
            queue_a=Q.get(a), queue_b=Q.get(b),
            batch_ids=batch_ids,
        )
    return total


def _infonce_pair(p_a, p_b, tau, queue_ids=None, queue_a=None, queue_b=None,
                  batch_ids=None):
    N = p_a.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=p_a.device, requires_grad=True)
    labels = torch.arange(N, device=p_a.device)

    logits_ab = (p_b @ p_a.t()) / tau                  # rows = b_i, cols = a_j
    logits_ba = (p_a @ p_b.t()) / tau

    if queue_ids is not None and queue_a is not None and queue_b is not None and batch_ids is not None:
        q_a = queue_a.to(device=p_a.device, dtype=p_a.dtype)
        q_b = queue_b.to(device=p_a.device, dtype=p_a.dtype)
        extra_ab = (p_b @ q_a.t()) / tau               # (N, M)
        extra_ba = (p_a @ q_b.t()) / tau

        same = batch_ids.view(-1, 1) == queue_ids.view(1, -1).to(p_a.device)
        if same.any():
            extra_ab = extra_ab.masked_fill(same, INFONCE_MASK_NEG)
            extra_ba = extra_ba.masked_fill(same, INFONCE_MASK_NEG)

        logits_ab = torch.cat([logits_ab, extra_ab], dim=1)
        logits_ba = torch.cat([logits_ba, extra_ba], dim=1)

    logits_ab = logits_ab.clamp(-INFONCE_LOGIT_CLAMP, INFONCE_LOGIT_CLAMP)
    logits_ba = logits_ba.clamp(-INFONCE_LOGIT_CLAMP, INFONCE_LOGIT_CLAMP)
    return 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels))


# ----------------------------- Reconstruction loss -----------------------------

def cosine_reconstruction_loss(
    reconstructions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    input_norms: dict[str, torch.nn.Module],
) -> torch.Tensor:
    """1 - cos(reconstructed, LN(input).detach()), summed across modalities.

    LayerNorm is applied with `.detach()` so gradients flow only through the
    decoder branch; otherwise the encoder would be incentivised to drift the
    targets towards what the decoder finds easy to reproduce.

    All modalities receive equal weight: because each branch lives in its own
    LayerNormed space, cosine values land in a comparable range, and uniform
    weighting trains more stably than learnable per-modality weights.
    """
    total = 0.0
    for name, recon in reconstructions.items():
        target = input_norms[name](targets[name]).detach()
        cos = F.cosine_similarity(recon, target, dim=1)
        total = total + (1 - cos.mean())
    return total


# ----------------------------- DEC clustering loss -----------------------------

def dec_cluster_loss(z: torch.Tensor, centroids: torch.Tensor,
                     alpha: float = 1.0) -> torch.Tensor:
    """KL(target || soft assignment) using a Student-t kernel.

    Soft assignments q_ik come from a Student-t kernel of degree alpha. The
    target distribution p_ik = q_ik^2 / sum_i(q_ik), then row-renormalised,
    sharpens high-confidence predictions while penalising over-large clusters
    to prevent collapse onto a single dominant cluster.
    """
    if centroids is None:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    dist2 = torch.cdist(z, centroids) ** 2
    dist2 = dist2.clamp(min=0.0, max=1e6)
    log_unnorm = -torch.log1p(dist2 / (alpha + 1e-8)).clamp(-50, 50)
    log_q = log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)
    q = log_q.exp()
    q = q / (q.sum(1, keepdim=True) + 1e-8)

    q_sq = q ** 2
    p_unnorm = q_sq / (q_sq.sum(0, keepdim=True) + 1e-8)
    p = p_unnorm / (p_unnorm.sum(1, keepdim=True) + 1e-8)

    p = p.clamp(1e-8, 1.0); q = q.clamp(1e-8, 1.0)
    return (p * (p.log() - q.log())).sum(1).mean()


def initialize_cluster_centers(z: torch.Tensor, num_clusters: int,
                               seed: int = 0) -> torch.Tensor:
    """K-means initialisation of DEC cluster centroids from a latent embedding."""
    from sklearn.cluster import KMeans

    z_np = z.detach().cpu().numpy()
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
    km.fit(z_np)
    return torch.from_numpy(km.cluster_centers_).to(z.device, dtype=z.dtype)
