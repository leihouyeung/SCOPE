# SCOPE

**S**ingle-**C**ell multim**O**dal s**P**atial int**E**gration --- a single
backbone for histology + spatial transcriptomics (+ optional spatial
proteomics) that produces one task-agnostic cell-state embedding.

| Modality | Pretrained encoder | Hugging Face id | Notes |
|---|---|---|---|
| histology | H0-mini ViT (Bioptimus) | `bioptimus/H0-mini` | Gated; set `HF_TOKEN` once |
| transcriptomics | NOVAE | `MICS-Lab/novae-human-0` | Public |
| proteomics | -- (arcsinh + clip + z-score) | -- | No external weights |

Worked end-to-end examples on full whole slides:

* [`examples/coad_bimodal_demo.ipynb`](examples/coad_bimodal_demo.ipynb) -- COAD (histology + transcriptomics)
* [`examples/rcc_trimodal_demo.ipynb`](examples/rcc_trimodal_demo.ipynb) -- RCC  (histology + transcriptomics + proteomics)

## Installation

```bash
conda env create -f environment.yaml
conda activate scope
pip install -e .
export HF_TOKEN=hf_xxxxxxxx           # accept the H0-mini licence first
```

## Histology aggregation modes

`preprocess_histology(..., mode=...)` exposes three ways to turn the H0-mini
ViT output into one 768-dim vector per cell:

| `mode` | What happens | When to use |
|---|---|---|
| `"resize56"`        | Crop 56x56 px around the centroid, bilinear-resize to 224x224, run the ViT, take the **CLS token**. | The cell occupies most of the receptive field; you want the model's holistic summary of the cell. |
| `"gaussian_input"`  | Crop 224x224 px, multiply pixel values by a 2D Gaussian centred on the cell (edges decay to 0), run the ViT, take the **mean of the 256 patch tokens**. | **Default.** Works without segmentation; centred cell dominates the receptive field while preserving local tissue context. |
| `"mask_token"`      | Crop 224x224 px, run the ViT unchanged, then take a weighted sum over the patch tokens where the weights are the per-token fraction of pixels labelled as the target cell in an instance segmentation mask. | You already have a segmentation. Pixel-accurate cell footprint. |

## Spatial graph

Cells are connected by a 2D **Delaunay triangulation** over their image-pixel
centroids. We drop the longest 1% of edges (`max_edge_percentile=99`) to
remove convex-hull "ghost" edges that bridge cells on opposite sides of the
slide. The resulting graph drives both the GAT layers and the cross-attention
distance bias.

## Quick start

```python
# --------------------------------------------------------------------------- #
# SCOPE quick start: bi-modal (histology + transcriptomics) on a single slide.
# The tri-modal version is identical except that you also call
# preprocess_proteomics(...) and add it as a third entry in `inputs`.
# --------------------------------------------------------------------------- #
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scope import (
    SCOPE, SCOPETrainer, TrainConfig,
    preprocess_histology, preprocess_transcriptomics,
)

# ---- CONFIG -- everything you might want to tweak lives here -------------- #
HISTO_CHECKPOINT = "bioptimus/H0-mini"          # any timm-loadable HF ViT
RNA_CHECKPOINT   = "MICS-Lab/novae-human-0"     # any NOVAE checkpoint
HF_TOKEN         = None                         # None -> use $HF_TOKEN env
HE_MODE          = "gaussian_input"             # resize56 | gaussian_input | mask_token
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

HE_PATH    = "data/COAD/HE.tif"                 # whole-slide RGB H&E
ADATA_PATH = "data/COAD/adata_fullres.h5ad"     # raw counts + obsm['spatial']
RUN_DIR    = "runs/coad"

EPOCHS, BATCH_SIZE, NUM_CLUSTERS = 200, 1024, 17
W_ALIGN, W_RECON, W_CLUSTER      = 1.0, 10.0, 5.0

# ---- 1. Load raw spatial transcriptomics + the H&E centroids -------------- #
# obsm['spatial'] holds 2-D centroids in *image-pixel* coordinates -- the same
# frame the H&E TIFF lives in, so cropping H&E patches per cell is just a
# NumPy slice.
adata     = ad.read_h5ad(ADATA_PATH)
coords_df = pd.DataFrame({
    "cell_id": adata.obs_names.astype(str),
    "x":       adata.obsm["spatial"][:, 0],
    "y":       adata.obsm["spatial"][:, 1],
})

# ---- 2. Histology -> H0-mini ViT -> (N, 768) ------------------------------ #
# H0-mini is frozen. The HE_MODE switch picks one of the three aggregation
# strategies described above. Patches are streamed in `chunk_size` cells at
# a time so 10^5--10^6 cells fit in <2 GB GPU.
he_feat = preprocess_histology(
    he_image   = HE_PATH,
    coords     = coords_df,
    mode       = HE_MODE,
    checkpoint = HISTO_CHECKPOINT,
    hf_token   = HF_TOKEN,
    device     = DEVICE,
)

# ---- 3. Transcriptomics -> NOVAE -> (N, 64) ------------------------------- #
# NOVAE expects raw counts + 2-D coordinates -- it owns its own normalisation
# (Pearson residuals + quantile scaling), so you must NOT pre-normalise or
# HVG-filter. Output is the 64-dim NOVAE latent.
rna_feat = preprocess_transcriptomics(
    adata      = adata,
    checkpoint = RNA_CHECKPOINT,
    device     = DEVICE,
)

# ---- 4. Train SCOPE ------------------------------------------------------- #
# SCOPE.from_inputs() infers each modality's input dim from the tensor shape,
# so you never have to keep dimensions in sync. With 2 modalities the
# adaptive gate is a scalar sigmoid; with >=3 it becomes a softmax simplex.
# Training mini-batches use Hilbert-curve ordering + 2-hop halo subgraphs
# so memory scales with batch size, not slide size. A single horizontal tqdm
# bar shows running loss as postfix.
inputs = {
    "histology":       torch.from_numpy(he_feat),
    "transcriptomics": torch.from_numpy(rna_feat),
}
coords = adata.obsm["spatial"].astype(np.float32)

model   = SCOPE.from_inputs(inputs)
trainer = SCOPETrainer(model, TrainConfig(
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    w_align=W_ALIGN, w_recon=W_RECON, w_cluster=W_CLUSTER,
    num_clusters=NUM_CLUSTERS, device=DEVICE,
))
trainer.fit(inputs, coords)

# ---- 5. Inference --------------------------------------------------------- #
# Mini-batched (Hilbert + halo) inference: scales to whole slides on one GPU.
# Output is a (N, 256) float32 embedding; feed it to scanpy/leiden/UMAP or
# any downstream task.
embedding = trainer.inference(inputs, coords)   # (N, 256)
```

## Tri-modal extension

Add one preprocess call and one more entry in `inputs`. Nothing else changes:

```python
from scope import preprocess_proteomics
pro_feat = preprocess_proteomics(adata=protein_adata)  # arcsinh+clip+z-score
inputs["proteomics"] = torch.from_numpy(pro_feat)
model = SCOPE.from_inputs(inputs)                      # now 3 modalities
```

## Saving and reloading

```python
torch.save({
    "state_dict":    model.state_dict(),
    "modality_dims": model.modality_dims,
}, "runs/coad/model.pt")

ckpt  = torch.load("runs/coad/model.pt", map_location="cpu")
model = SCOPE(modality_dims=ckpt["modality_dims"])
model.load_state_dict(ckpt["state_dict"])
```

## Reproducibility

Seed is 42 (internal). Reference stack: CUDA 12.1, PyTorch 2.1, PyG 2.5.

## License

MIT.
