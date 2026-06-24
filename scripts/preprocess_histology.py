#!/usr/bin/env python3
"""CLI wrapper around ``scope.preprocess.preprocess_histology``.

For interactive (Jupyter) use, prefer:
    from scope import preprocess_histology
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.preprocess import PIXEL_SIZE_UM, preprocess_histology


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--he-image", type=Path, required=True,
                   help="Whole-slide H&E (TIFF or PNG, RGB)")
    p.add_argument("--coords", type=Path, required=True,
                   help="CSV with columns cell_id,x,y (centroids in image pixels)")
    p.add_argument("--mask", type=Path, default=None,
                   help="Optional instance-segmentation TIFF for mask-weighted pooling")
    p.add_argument("--checkpoint", type=str, default="bioptimus/H0-mini",
                   help="Hugging Face Hub id of the H0-mini checkpoint")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HF token (defaults to HF_TOKEN env var)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Cells per streaming chunk (memory <-> throughput trade-off)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    for path in (args.he_image, args.coords):
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")
    if args.mask is not None and not args.mask.exists():
        raise SystemExit(f"--mask file not found: {args.mask}")

    coords = pd.read_csv(args.coords)
    feat = preprocess_histology(
        he_image=args.he_image,
        coords=coords,
        mask=args.mask,
        checkpoint=args.checkpoint,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    out = ad.AnnData(
        X=feat,
        obs={"cell_id": coords["cell_id"].astype(str).values},
        obsm={"spatial": coords[["x", "y"]].to_numpy(dtype=np.float32) * PIXEL_SIZE_UM},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output)
    print(f"[save] {args.output}  shape={feat.shape}")


if __name__ == "__main__":
    main()
