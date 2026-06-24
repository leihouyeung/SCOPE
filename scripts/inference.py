#!/usr/bin/env python3
"""Run SCOPE inference on new data using a trained checkpoint.

Example:
    python scripts/inference.py \\
        --checkpoint runs/rcc/model.pt \\
        --histology new_histology.h5ad \\
        --transcriptomics new_rna.h5ad \\
        --proteomics new_protein.h5ad \\
        --output runs/rcc/new_embedding.h5ad
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to model.pt saved by scripts/train.py")
    for name in MODALITY_REGISTRY:
        p.add_argument(f"--{name}", type=Path, default=None,
                       help=f"h5ad path for the {name} modality")
        p.add_argument(f"--{name}-key", type=str, default="X")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load(path: Path, key: str):
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    a = ad.read_h5ad(path)
    if key == "X":
        feat = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    else:
        feat = np.asarray(a.obsm[key])
    return a, feat.astype(np.float32)


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"--checkpoint file not found: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except Exception as e:
        raise SystemExit(f"Failed to load checkpoint {args.checkpoint}: {e}") from e

    required = {"state_dict", "modality_dims"}
    missing = required - set(ckpt)
    if missing:
        raise SystemExit(
            f"Checkpoint is missing required keys: {sorted(missing)}. "
            f"The file may be corrupted or from an incompatible SCOPE version."
        )

    model = SCOPE(modality_dims=ckpt["modality_dims"])
    try:
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        raise SystemExit(
            f"State dict does not match the architecture from the checkpoint "
            f"metadata. The checkpoint may have been saved with a different "
            f"SCOPE version. Original error: {e}"
        ) from e

    needed = set(ckpt["modality_dims"].keys())
    sources = {}
    for name in MODALITY_REGISTRY:
        path = getattr(args, name.replace("-", "_"))
        if path is None:
            continue
        if name not in needed:
            print(f"[warn] {name} provided but not in checkpoint; ignored.")
            continue
        key = getattr(args, f"{name.replace('-', '_')}_key")
        sources[name] = _load(path, key)
    if needed != set(sources.keys()):
        raise SystemExit(f"Modality mismatch: checkpoint expects {sorted(needed)}, "
                         f"got {sorted(sources.keys())}.")

    a0 = next(iter(sources.values()))[0]
    coords = np.asarray(a0.obsm["spatial"], dtype=np.float32)
    inputs = {n: torch.from_numpy(f) for n, (_, f) in sources.items()}

    config = TrainConfig(batch_size=args.batch_size, device=args.device)
    trainer = SCOPETrainer(model, config)
    embedding = trainer.inference(inputs, coords)

    out = ad.AnnData(X=embedding, obs=a0.obs.copy(), obsm={"spatial": coords})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output)
    print(f"[save] {args.output}  shape={embedding.shape}")


if __name__ == "__main__":
    main()
