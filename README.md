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

adata     = ad.read_h5ad(ADATA_PATH)
coords_df = pd.DataFrame({
    "cell_id": adata.obs_names.astype(str),
    "x":       adata.obsm["spatial"][:, 0],
    "y":       adata.obsm["spatial"][:, 1],
})

# ---- 2. Histology -> H0-mini ViT -> (N, 768) ------------------------------ #

he_feat = preprocess_histology(
    he_image   = HE_PATH,
    coords     = coords_df,
    mode       = HE_MODE,
    checkpoint = HISTO_CHECKPOINT,
    hf_token   = HF_TOKEN,
    device     = DEVICE,
)

# ---- 3. Transcriptomics -> NOVAE -> (N, 64) ------------------------------- #

rna_feat = preprocess_transcriptomics(
    adata      = adata,
    checkpoint = RNA_CHECKPOINT,
    device     = DEVICE,
)

# ---- 4. Train SCOPE ------------------------------------------------------- #

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



## License

MIT.
