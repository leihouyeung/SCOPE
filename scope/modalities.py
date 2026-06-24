"""Modality registry.

SCOPE supports an arbitrary number of input modalities through a dict-based
configuration. To keep the supported modalities discoverable and to provide
a single source of truth for their preprocessing recipes, expected feature
dimensions and CLI surfaces, we register them here.

Adding a new modality:
    1. Implement `scripts/preprocess_<name>.py` that writes an h5ad with
       per-cell features in `.X` and 2D coordinates in `.obsm["spatial"]`.
    2. Append a `ModalityInfo` entry to `MODALITY_REGISTRY` below.
    3. Update the corresponding `scripts/train.py` flag (e.g. `--<name>`).

Note: the names used here are the semantic identifiers that appear in
checkpoints (`SCOPE.modality_names`), in CLI flags and in figures. Keeping
them in sync across the codebase removes the ambiguity of short tokens
such as `img` or `pro`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModalityInfo:
    """Metadata for a registered modality."""

    name: str                          # canonical name used throughout SCOPE
    preprocess_script: str             # entry point under scripts/
    default_dim: Optional[int]         # expected per-cell feature dim (None if variable)
    short_alias: str                   # legacy short name kept for backward compatibility
    description: str                   # one-line description for help text and README


MODALITY_REGISTRY: dict[str, ModalityInfo] = {
    "histology": ModalityInfo(
        name="histology",
        preprocess_script="scripts/preprocess_histology.py",
        default_dim=768,
        short_alias="img",
        description=(
            "H&E cell-anchored embedding from the H0-mini ViT. Patches of "
            "224x224 are aggregated into a single 768-d vector per cell using "
            "either a Gaussian or instance-mask weighting on the 14x14 token grid."
        ),
    ),
    "transcriptomics": ModalityInfo(
        name="transcriptomics",
        preprocess_script="scripts/preprocess_transcriptomics.py",
        default_dim=64,
        short_alias="rna",
        description=(
            "Spatial transcriptomics latent from the NOVAE foundation model. "
            "Raw per-cell counts together with 2D coordinates are fed into the "
            "frozen MICS-Lab/novae-human-0 encoder, yielding a 64-d vector."
        ),
    ),
    "proteomics": ModalityInfo(
        name="proteomics",
        preprocess_script="scripts/preprocess_proteomics.py",
        default_dim=None,
        short_alias="pro",
        description=(
            "Spatial proteomics channels after arcsinh(x/5), per-channel 1st-99th "
            "percentile clipping and per-channel z-scoring. The normalised channels "
            "are fed directly to the SCOPE encoder MLP without any factor analysis."
        ),
    ),
}


def get_modality(name: str) -> ModalityInfo:
    """Resolve a modality name or its short alias to a ModalityInfo entry."""
    if name in MODALITY_REGISTRY:
        return MODALITY_REGISTRY[name]
    for info in MODALITY_REGISTRY.values():
        if info.short_alias == name:
            return info
    raise KeyError(
        f"Unknown modality '{name}'. Registered modalities: "
        f"{sorted(MODALITY_REGISTRY.keys())}"
    )


def list_modalities() -> list[str]:
    """Return the canonical names of all registered modalities."""
    return sorted(MODALITY_REGISTRY.keys())
