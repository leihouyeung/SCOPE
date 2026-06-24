#!/usr/bin/env python3
"""CLI wrapper around ``scope.preprocess.preprocess_proteomics``.

For interactive (Jupyter) use, prefer:
    from scope import preprocess_proteomics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.preprocess import preprocess_proteomics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="h5ad with raw channel intensities in .X")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--section-key", type=str, default=None,
                   help="Optional .obs column for within-section normalisation")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"--input file not found: {args.input}")

    adata = ad.read_h5ad(args.input)
    feat = preprocess_proteomics(adata=adata, section_key=args.section_key)
    adata.X = feat

    obs = adata.obs.copy()
    if "cell_id" not in obs.columns:
        obs.insert(0, "cell_id", adata.obs_names.astype(str).values)
    adata.obs = obs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    scope_note = "within-section" if args.section_key else "global"
    print(f"[save] {args.output}  shape={feat.shape}  ({scope_note} z-score)")


if __name__ == "__main__":
    main()
