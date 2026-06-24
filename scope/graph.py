"""Spatial graph construction and Hilbert-ordered batching.

Provides:
    build_delaunay_graph(coords)       -> edge_index with self-loops
    hilbert_sort_indices(coords)       -> permutation for compact mini-batches
    expand_core_halo(adj, core, hops)  -> add k-hop neighbors around a batch
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay


# ----------------------------- Delaunay graph -----------------------------

def build_delaunay_graph(coords: np.ndarray, add_self_loops: bool = True,
                         knn_fallback_k: int = 6,
                         max_edge_percentile: float = 99.0,
                         ) -> tuple[torch.Tensor, float]:
    """Build a Delaunay triangulation graph over 2D cell centroids.

    Args:
        coords: (N, 2) float array of cell positions (in micrometres).
        add_self_loops: whether to add (i, i) edges so each cell can preserve
            its own modality-specific signal during message passing.
        knn_fallback_k: number of neighbours to use if Delaunay fails on
            degenerate (e.g., collinear) point sets.
        max_edge_percentile: drop edges whose length exceeds this percentile
            of the non-self-loop edge-length distribution. Removes the
            convex-hull "ghost" edges that connect cells on opposite sides
            of the slide. Set to ``None`` or >=100 to keep all edges.

    Returns:
        edge_index: (2, E) int64 tensor of directed edges.
        median_edge_length: physical-scale anchor used by distance-bias modules.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2); got {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError("coords contain NaN or Inf values.")

    if coords.shape[0] < 4:
        # Fall back to all-pairs for tiny tiles.
        idx = np.arange(coords.shape[0])
        edges = np.stack(np.meshgrid(idx, idx), -1).reshape(-1, 2)
    else:
        try:
            from scipy.spatial.qhull import QhullError  # local import
        except ImportError:
            QhullError = Exception
        try:
            tri = Delaunay(coords)
            s = tri.simplices
            edges = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]], axis=0)
            edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
            edges = np.unique(edges, axis=0)
        except QhullError:
            # Degenerate point set (e.g., collinear); fall back to k-NN.
            from sklearn.neighbors import NearestNeighbors
            k = min(knn_fallback_k + 1, coords.shape[0])
            nn = NearestNeighbors(n_neighbors=k).fit(coords)
            _, idx_nn = nn.kneighbors(coords)
            src = np.repeat(np.arange(coords.shape[0]), k - 1)
            dst = idx_nn[:, 1:].reshape(-1)
            edges = np.stack([src, dst], axis=1)
            edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
            edges = np.unique(edges, axis=0)

    # Trim convex-hull "ghost" edges by edge-length percentile.
    if max_edge_percentile is not None and max_edge_percentile < 100:
        diff_pre = coords[edges[:, 0]] - coords[edges[:, 1]]
        lengths_pre = np.linalg.norm(diff_pre, axis=1)
        nz = lengths_pre > 0
        if nz.any():
            cutoff = float(np.percentile(lengths_pre[nz], max_edge_percentile))
            edges = edges[lengths_pre <= cutoff]

    if add_self_loops:
        loops = np.stack([np.arange(coords.shape[0])] * 2, axis=1)
        edges = np.concatenate([edges, loops], axis=0)

    # Median edge length (excluding self-loops) is the per-section physical scale.
    diff = coords[edges[:, 0]] - coords[edges[:, 1]]
    lengths = np.linalg.norm(diff, axis=1)
    median = float(np.median(lengths[lengths > 0])) if (lengths > 0).any() else 1.0

    edge_index = torch.from_numpy(edges.T).long().contiguous()
    return edge_index, median


# ----------------------------- Hilbert ordering -----------------------------

def hilbert_sort_indices(coords: np.ndarray, order: int = 10) -> np.ndarray:
    """Return a permutation of cells along a Hilbert space-filling curve.

    Mini-batches consumed in this order occupy compact physical regions, which
    preserves local graph structure and reduces boundary artefacts compared
    with random shuffling.
    """
    n = coords.shape[0]
    coords = np.asarray(coords, dtype=np.float64)
    # Normalise to integer grid of side 2**order.
    side = 1 << order
    mn, mx = coords.min(0), coords.max(0)
    rng = np.maximum(mx - mn, 1e-9)
    xy = np.floor((coords - mn) / rng * (side - 1)).astype(np.int64)
    keys = np.array([_xy_to_hilbert(int(x), int(y), order) for x, y in xy], dtype=np.int64)
    return np.argsort(keys, kind="stable")


def _xy_to_hilbert(x: int, y: int, order: int) -> int:
    """Map (x, y) on a 2**order grid to a 1-D Hilbert index."""
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        s >>= 1
    return d


# ----------------------------- Halo expansion -----------------------------

def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> csr_matrix:
    """Build a sparse undirected adjacency matrix for halo expansion."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    mask = src != dst
    src, dst = src[mask], dst[mask]
    data = np.ones(src.size, dtype=np.uint8)
    A = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    A = A + A.T
    A.data[:] = 1
    return A


def expand_core_halo(adj: csr_matrix, core_nodes: np.ndarray, hops: int) -> np.ndarray:
    """Expand a core batch by `hops` BFS steps so first-order neighbours are
    available even at section boundaries. Returns sorted unique node ids.
    """
    if hops <= 0:
        return np.unique(core_nodes)
    frontier = set(int(i) for i in core_nodes)
    visited = set(frontier)
    indices, indptr = adj.indices, adj.indptr
    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            new_frontier.update(int(j) for j in indices[indptr[node]:indptr[node + 1]])
        new_frontier -= visited
        if not new_frontier:
            break
        visited.update(new_frontier)
        frontier = new_frontier
    return np.array(sorted(visited), dtype=np.int64)
