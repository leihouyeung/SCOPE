#!/usr/bin/env python3
"""Train SCOPE on bi- or tri-modal spatial omics data.

The script reads one .h5ad per modality, aligns cells across modalities by
their `cell_id` (or `obs_names` fallback), and trains a SCOPE model. The
number of modalities is inferred automatically from the arguments
provided: supply any two of {--histology, --transcriptomics, --proteomics}
for bi-modal, all three for tri-modal.

Example -- tri-modal renal cell carcinoma
-----------------------------------------
    python scripts/train.py \\
        --histology histology.h5ad \\
        --transcriptomics transcriptomics.h5ad \\
        --proteomics proteomics.h5ad \\
        --output runs/rcc/ --epochs 200 \\
        --w-align 1 --w-recon 10 --w-cluster 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope import SCOPE, SCOPETrainer, TrainConfig
from scope.modalities import MODALITY_REGISTRY


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # ---- One flag per registered modality, plus a matching --<name>-key.
    for name in MODALITY_REGISTRY:
        p.add_argument(f"--{name}", type=Path, default=None,
                       help=f"h5ad path for the {name} modality "
                            f"(produced by {MODALITY_REGISTRY[name].preprocess_script})")
        p.add_argument(f"--{name}-key", type=str, default="X",
                       help=f".obsm key (or 'X') containing the {name} feature matrix")

    # ---- Output.
    p.add_argument("--output", type=Path, required=True)

    # ---- Optimisation (the only user-tunable training knobs).
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)

    # ---- Loss.
    p.add_argument("--w-align", type=float, default=1.0)
    p.add_argument("--w-recon", type=float, default=10.0)
    p.add_argument("--w-cluster", type=float, default=5.0)
    p.add_argument("--num-clusters", type=int, default=12)
    p.add_argument("--xbm-size", type=int, default=65536)

    # ---- Hardware.
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ----------------------------- Data helpers -----------------------------

def _load(path: Path, key: str) -> tuple[ad.AnnData, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    try:
        a = ad.read_h5ad(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e
    if key == "X":
        feat = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    else:
        if key not in a.obsm:
            raise KeyError(
                f"obsm key '{key}' missing in {path}. "
                f"Available keys: {sorted(a.obsm.keys())}"
            )
        feat = np.asarray(a.obsm[key])
    feat = feat.astype(np.float32)
    if feat.ndim != 2:
        raise ValueError(
            f"{path} feature matrix has shape {feat.shape}; expected 2D."
        )
    if not np.isfinite(feat).all():
        n_bad = int(np.logical_not(np.isfinite(feat)).sum())
        raise ValueError(
            f"{path} contains {n_bad} non-finite values. "
            f"Run the corresponding preprocess script before training."
        )
    return a, feat


def _cell_ids(adata: ad.AnnData) -> np.ndarray:
    col = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs else adata.obs_names.astype(str)
    return col.values


def _align(adatas: dict[str, ad.AnnData]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Intersect by cell_id and reorder; coordinates from the first modality."""
    per_modality_counts = {n: len(_cell_ids(a)) for n, a in adatas.items()}
    print(f"[align] cell counts per modality: {per_modality_counts}")
    ids_set = None
    for a in adatas.values():
        ids_set = set(_cell_ids(a)) if ids_set is None else ids_set & set(_cell_ids(a))
    shared = sorted(ids_set)
    if not shared:
        names = list(adatas)
        raise SystemExit(
            f"Modalities {names} have no overlapping cell ids. "
            f"Check the 'cell_id' obs column or obs_names across the per-modality "
            f"h5ad files (see preprocess_*.py output)."
        )
    if len(shared) < 32:
        raise SystemExit(
            f"Only {len(shared)} cells are shared across modalities; "
            f"SCOPE requires at least a few dozen cells for training. "
            f"Check the modality-alignment step."
        )

    a0 = next(iter(adatas.values()))
    if "spatial" not in a0.obsm:
        raise SystemExit(
            f"The first modality ({next(iter(adatas))}) has no obsm['spatial']. "
            f"SCOPE requires 2D coordinates."
        )
    pos0 = {cid: i for i, cid in enumerate(_cell_ids(a0))}
    order_each = {name: np.array([{cid: i for i, cid in enumerate(_cell_ids(a))}[c] for c in shared])
                  for name, a in adatas.items()}
    coords = np.asarray(a0.obsm["spatial"], dtype=np.float32)[np.array([pos0[c] for c in shared])]
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise SystemExit(
            f"obsm['spatial'] must be a 2D coordinate matrix (N, 2); "
            f"got shape {coords.shape}."
        )
    if not np.isfinite(coords).all():
        raise SystemExit("obsm['spatial'] contains NaN or Inf.")
    return shared, coords, order_each


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Validate CUDA availability when requested.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] device='{args.device}' requested but CUDA is unavailable; "
              f"falling back to CPU.")
        args.device = "cpu"

    # ---- Load modalities.
    raw: dict[str, ad.AnnData] = {}
    feats: dict[str, np.ndarray] = {}
    for name in MODALITY_REGISTRY:
        path = getattr(args, name.replace("-", "_"))
        if path is None:
            continue
        key = getattr(args, f"{name.replace('-', '_')}_key")
        a, f = _load(path, key)
        if "spatial" not in a.obsm:
            raise SystemExit(f"{name} h5ad must contain obsm['spatial'].")
        raw[name] = a
        feats[name] = f

    if len(raw) < 2:
        raise SystemExit("Please supply at least two of "
                         + " / ".join(f"--{n}" for n in MODALITY_REGISTRY))

    shared, coords, order_each = _align(raw)
    print(f"[data] Aligned {len(shared)} cells across {len(raw)} modalities: "
          f"{', '.join(raw.keys())}.")

    aligned = {n: feats[n][order_each[n]] for n in raw}
    inputs = {n: torch.from_numpy(x).float() for n, x in aligned.items()}

    # ---- Build the model. Per-modality dims are inferred from the data.
    model = SCOPE.from_inputs(inputs)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters.")

    # ---- Train.
    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        w_align=args.w_align, w_recon=args.w_recon, w_cluster=args.w_cluster,
        num_clusters=args.num_clusters, xbm_size=args.xbm_size, device=args.device,
    )
    trainer = SCOPETrainer(model, config)
    trainer.fit(inputs, coords)

    # ---- Save embedding + model.
    embedding = trainer.inference(inputs, coords)
    obs = {"cell_id": shared}
    out = ad.AnnData(X=embedding, obs=obs, obsm={"spatial": coords})
    out.write_h5ad(args.output / "embedding.h5ad")

    torch.save({
        "state_dict": model.state_dict(),
        "modality_dims": model.modality_dims,
        "modality_names": list(model.modality_dims.keys()),
    }, args.output / "model.pt")
    with (args.output / "config.json").open("w") as f:
        json.dump(vars(args) | {"modality_dims": model.modality_dims},
                  f, indent=2, default=str)
    print(f"[save] embedding -> {args.output / 'embedding.h5ad'}")
    print(f"[save] model     -> {args.output / 'model.pt'}")


if __name__ == "__main__":
    main()
