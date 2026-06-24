#!/usr/bin/env python3
"""CLI wrapper around ``scope.preprocess.preprocess_transcriptomics``.

For interactive (Jupyter) use, prefer:
    from scope import preprocess_transcriptomics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.preprocess import preprocess_transcriptomics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="h5ad with raw counts in .X and .obsm['spatial']")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, default="MICS-Lab/novae-human-0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"--input file not found: {args.input}")

    adata = ad.read_h5ad(args.input)
    emb = preprocess_transcriptomics(
        adata=adata,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    obs = adata.obs.copy()
    if "cell_id" not in obs.columns:
        obs.insert(0, "cell_id", adata.obs_names.astype(str).values)
    out = ad.AnnData(X=emb, obs=obs, obsm={"spatial": np.asarray(adata.obsm["spatial"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output)
    print(f"[save] {args.output}  shape={emb.shape}  (NOVAE latent)")


if __name__ == "__main__":
    main()
