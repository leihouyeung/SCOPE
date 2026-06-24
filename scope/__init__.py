"""SCOPE: Single-Cell multimOdal sPatial intEgration.

A unified framework for integrating histology, spatial transcriptomics and
spatial proteomics into a single task-agnostic cell-state representation.

The model supports any subset of the registered modalities:

    >>> from scope import SCOPE, SCOPETrainer, TrainConfig, MODALITY_REGISTRY
    >>> sorted(MODALITY_REGISTRY)
    ['histology', 'proteomics', 'transcriptomics']
    >>> model = SCOPE(modality_dims={
    ...     "histology": 768,
    ...     "transcriptomics": 64,
    ...     "proteomics": 27,
    ... })
    >>> trainer = SCOPETrainer(model, TrainConfig(epochs=200))
    >>> trainer.fit(inputs, coords)
    >>> embedding = trainer.inference(inputs, coords)
"""
from .model import SCOPE
from .trainer import SCOPETrainer, TrainConfig
from .graph import build_delaunay_graph, hilbert_sort_indices
from .modalities import MODALITY_REGISTRY, ModalityInfo, get_modality, list_modalities
from .preprocess import (
    H0MiniEncoder,
    preprocess_histology,
    preprocess_transcriptomics,
    preprocess_proteomics,
)

__version__ = "1.0.0"
__all__ = [
    "SCOPE", "SCOPETrainer", "TrainConfig",
    "build_delaunay_graph", "hilbert_sort_indices",
    "MODALITY_REGISTRY", "ModalityInfo", "get_modality", "list_modalities",
    "H0MiniEncoder", "preprocess_histology",
    "preprocess_transcriptomics", "preprocess_proteomics",
]
