"""Importable preprocessing API for SCOPE.

Three functions, one per modality, all of which return either a NumPy
array of per-cell features or an AnnData object ready for training:

    preprocess_histology(...)        -> (N, 768) np.float32
    preprocess_transcriptomics(...)  -> (N, 64) np.float32
    preprocess_proteomics(...)       -> (N, D_pro) np.float32

The histology pipeline streams patches in chunks, so it scales to whole
slides (hundreds of thousands of cells) without pre-allocating the full
patch tensor.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm   # ipywidget bar in notebooks, terminal bar in CLI

# Whole-slide H&E images routinely exceed PIL's default 178 Mpx safety cap
# (e.g. COAD Xenium HE.tif is ~1.78 Gpx). The cap is a DoS guard for
# untrusted inputs; histology pipelines own their data, so disable it.
Image.MAX_IMAGE_PIXELS = None


# -------------------- Fixed-default constants (no CLI surface) --------------------

# Histology: 224x224 patch -> H0-mini ViT -> spatial token grid.
# Gaussian aggregation: sigma=10 micrometres (about one nuclear diameter),
# assuming a 0.5 um/pixel H&E acquisition (Xenium / CODEX-CARTANA standard).
PATCH_SIZE = 224
SIGMA_UM = 10.0
PIXEL_SIZE_UM = 0.5

# Transcriptomics: NOVAE input graph radius.
SPATIAL_RADIUS = 80.0

# Proteomics: arcsinh(x/5) + per-channel 1st-99th percentile clip + z-score.
COFACTOR = 5.0
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


# ----------------------------- H0-mini encoder -----------------------------

class H0MiniEncoder:
    """Thin wrapper around the H0-mini ViT for spatial-token extraction.

    Loads the model via `timm.create_model('hf-hub:bioptimus/H0-mini', ...)`,
    streams batches of (B, 224, 224, 3) uint8 patches, and returns the
    (B, n_tokens, 768) spatial patch tokens (CLS / register prefixes stripped).

    Allocate the encoder ONCE per session; the model load + cuDNN warmup
    costs about 10 seconds on the first call.
    """

    def __init__(self, checkpoint: str = "bioptimus/H0-mini",
                 device: Optional[str] = None, hf_token: Optional[str] = None,
                 mixed_precision: bool = True):
        try:
            import timm
            from timm.data import resolve_data_config
        except ImportError as e:
            raise SystemExit(
                "Required packages missing for H0-mini. Run: "
                "`pip install timm huggingface_hub`."
            ) from e

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mixed_precision = mixed_precision

        token = (hf_token or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGINGFACE_TOKEN")
                 or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
            except Exception as e:
                print(f"[warn] huggingface_hub login failed ({e}); will rely on "
                      f"the model cache if available.")

        source = checkpoint if str(checkpoint).startswith("hf-hub:") else f"hf-hub:{checkpoint}"
        try:
            base = timm.create_model(
                source, pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            ).to(self.device).eval()
        except Exception as e:
            raise SystemExit(
                f"Failed to load H0-mini checkpoint '{checkpoint}'. "
                f"If this is the first run, set the HF_TOKEN environment "
                f"variable to a Hugging Face token that has accepted the "
                f"bioptimus/H0-mini licence, then retry. Original error: {e}"
            ) from e
        self.model = base
        self.n_prefix = int(base.num_prefix_tokens)

        data_config = resolve_data_config(base.pretrained_cfg, model=base)
        self.mean = torch.tensor(data_config.get("mean", (0.485, 0.456, 0.406)),
                                 dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(data_config.get("std", (0.229, 0.224, 0.225)),
                                dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.patch_px = int(base.patch_embed.patch_size[0])

    @torch.inference_mode()
    def encode(self, patches: np.ndarray, batch_size: int = 64,
               include_prefix: bool = False) -> np.ndarray:
        """``(N, H, W, 3)`` uint8 patches -> ViT token features.

        Returns ``(N, n_patch_tokens, 768)`` by default; when
        ``include_prefix=True``, returns ``(N, n_prefix + n_patch_tokens, 768)``
        so the caller can pick the CLS token at position 0.
        """
        gh = patches.shape[1] // self.patch_px
        gw = patches.shape[2] // self.patch_px
        if gh * self.patch_px != patches.shape[1] or gw * self.patch_px != patches.shape[2]:
            raise ValueError(f"Patch H, W must be multiples of {self.patch_px}.")
        ctx = (torch.autocast(device_type=self.device.type, dtype=torch.float16)
               if self.mixed_precision else nullcontext())
        feats = []
        with ctx:
            for start in range(0, patches.shape[0], batch_size):
                batch = patches[start:start + batch_size]
                x = (torch.from_numpy(batch).to(self.device, non_blocking=True)
                     .float().div_(255.0).permute(0, 3, 1, 2).contiguous())
                x = (x - self.mean) / self.std
                out = self.model(x)
                if not include_prefix:
                    out = out[:, self.n_prefix:, :]
                feats.append(out.float().cpu())
        return torch.cat(feats, dim=0).numpy().astype(np.float32)


# ----------------------------- Aggregation rules -----------------------------

# Mode 1 (`resize56`) crops this many image pixels around the cell centre and
# resizes the result to PATCH_SIZE x PATCH_SIZE before feeding to the ViT.
SMALL_CROP_PX = 56


def _input_gaussian_mask(size: int, sigma_pix: float) -> np.ndarray:
    """Centred 2D Gaussian on a ``size x size`` *image-pixel* grid.

    Used by Mode 2 (`gaussian_input`) to multiply the input patch so that the
    cell at the centre dominates the ViT's receptive field. Peak is 1, drops
    off radially with sigma=``sigma_pix``. Shape ``(size, size)`` float32.
    """
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma_pix ** 2))
    return g.astype(np.float32)


def _mask_token_weights(mask_patch: np.ndarray, target_id: int,
                        grid_size: int, patch_px: int) -> np.ndarray:
    """Fraction of ``target_id`` pixels per token receptive field."""
    bin_mask = (mask_patch == target_id).astype(np.float32)
    grid = bin_mask.reshape(grid_size, patch_px, grid_size, patch_px).sum(axis=(1, 3))
    grid = grid.reshape(-1)
    s = grid.sum()
    return grid / s if s > 0 else None


def _crop(image: np.ndarray, cx: int, cy: int, size: int,
          fill_val: int | float = 0) -> np.ndarray:
    """Crop a size x size patch centred at (cx, cy), zero-padding at borders."""
    h, w = image.shape[:2]
    half = size // 2
    out_shape = (size, size, image.shape[2]) if image.ndim == 3 else (size, size)
    out = np.full(out_shape, fill_val, dtype=image.dtype)
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    dx, dy = x0 - (cx - half), y0 - (cy - half)
    out[dy:dy + (y1 - y0), dx:dx + (x1 - x0)] = image[y0:y1, x0:x1]
    return out


def _resize_rgb_uint8(arr: np.ndarray, target: int) -> np.ndarray:
    """Bilinear-resize an ``(H, W, 3)`` uint8 patch to ``target x target``."""
    return np.asarray(Image.fromarray(arr).resize((target, target),
                                                  Image.Resampling.BILINEAR),
                      dtype=np.uint8)


# ----------------------------- Public API: histology -----------------------------

def preprocess_histology(
    he_image: Union[np.ndarray, Path, str],
    coords: pd.DataFrame,
    mask: Union[np.ndarray, Path, str, None] = None,
    mode: str = "gaussian_input",
    checkpoint: str = "bioptimus/H0-mini",
    hf_token: Optional[str] = None,
    batch_size: int = 64,
    chunk_size: int = 256,
    device: Optional[str] = None,
    show_progress: bool = True,
    encoder: Optional[H0MiniEncoder] = None,
) -> np.ndarray:
    """Encode H&E patches into 768-dim cell-anchored embeddings.

    Patches are streamed in ``chunk_size`` cells at a time so the function
    scales to whole-slide datasets with hundreds of thousands of cells.

    Three aggregation modes are supported:

    * ``"resize56"``  -- crop a 56 x 56 px patch around the centroid, bilinear
      resize to 224 x 224, run the ViT, and take the **CLS token** as the cell
      embedding. Best when the cell occupies most of the receptive field.

    * ``"gaussian_input"`` -- crop 224 x 224, multiply the *pixel values* by a
      2D Gaussian centred on the cell so distant regions decay toward 0, run
      the ViT, and take the **mean** of the 256 spatial patch tokens. This is
      the default; works without segmentation.

    * ``"mask_token"`` -- crop 224 x 224 and run the ViT unchanged, then
      take a weighted sum over patch tokens where each weight is the fraction
      of pixels in that token receptive field that belong to the target cell
      (read from ``mask``). Falls back to ``"gaussian_input"`` for cells with
      no mask coverage.

    Args:
        he_image: RGB H&E image (NumPy ``(H, W, 3)`` uint8) or a path to one.
        coords: DataFrame with columns ``cell_id, x, y`` (centroids in image pixels).
        mask: optional instance-segmentation array aligned to the H&E. Pixel
            values are cell ids (0 = background). Required when
            ``mode='mask_token'``; ignored otherwise.
        mode: one of ``"resize56"``, ``"gaussian_input"``, ``"mask_token"``.
        checkpoint: Hugging Face Hub id of the H0-mini ViT.
        hf_token: HF access token (defaults to the ``HF_TOKEN`` env var).
        batch_size: H0-mini inference batch size.
        chunk_size: cells per streaming chunk. Memory use scales as
            ``chunk_size * 224 * 224 * 3 = ~38 MB`` per chunk for the default.
        device: ``cuda`` / ``cuda:0`` / ``cpu`` (defaults to cuda when available).
        show_progress: display a tqdm progress bar.
        encoder: optional pre-loaded ``H0MiniEncoder``. Pass this to avoid
            re-creating the model when running the pipeline repeatedly.

    Returns:
        ``(N, 768)`` ``np.float32`` array of per-cell H&E embeddings.
    """
    if mode not in {"resize56", "gaussian_input", "mask_token"}:
        raise ValueError(f"mode must be one of resize56|gaussian_input|"
                         f"mask_token; got {mode!r}.")
    if mode == "mask_token" and mask is None:
        raise ValueError("mode='mask_token' requires a `mask` argument.")

    if isinstance(he_image, (str, Path)):
        he_image = np.asarray(Image.open(he_image).convert("RGB"), dtype=np.uint8)
    if mask is not None and isinstance(mask, (str, Path)):
        try:
            import tifffile
            mask = tifffile.imread(mask)
        except ImportError:
            mask = np.asarray(Image.open(mask))
        mask = np.asarray(mask).astype(np.int32)

    for col in ("cell_id", "x", "y"):
        if col not in coords.columns:
            raise ValueError(f"coords is missing required column '{col}'.")

    if encoder is None:
        encoder = H0MiniEncoder(checkpoint=checkpoint, device=device,
                                hf_token=hf_token, mixed_precision=True)

    grid = PATCH_SIZE // encoder.patch_px
    sigma_pix = SIGMA_UM / PIXEL_SIZE_UM
    gauss_pixel = _input_gaussian_mask(PATCH_SIZE, sigma_pix)[..., None]  # (224,224,1)

    n_cells = len(coords)
    output = np.zeros((n_cells, 768), dtype=np.float32)
    n_fallback = 0
    xs = coords["x"].to_numpy(dtype=np.float32)
    ys = coords["y"].to_numpy(dtype=np.float32)
    cids = (coords["cell_id"].to_numpy() if mode == "mask_token" else None)

    iterator = range(0, n_cells, chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="H&E preprocessing", unit="chunk")

    patches = np.empty((chunk_size, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    mask_patches = (np.empty((chunk_size, PATCH_SIZE, PATCH_SIZE), dtype=np.int32)
                    if mode == "mask_token" else None)

    for start in iterator:
        end = min(start + chunk_size, n_cells)
        m = end - start
        for i in range(m):
            cx, cy = int(round(xs[start + i])), int(round(ys[start + i]))
            if mode == "resize56":
                small = _crop(he_image, cx, cy, SMALL_CROP_PX)
                patches[i] = _resize_rgb_uint8(small, PATCH_SIZE)
            else:
                raw = _crop(he_image, cx, cy, PATCH_SIZE)
                if mode == "gaussian_input":
                    raw = (raw.astype(np.float32) * gauss_pixel).clip(0, 255).astype(np.uint8)
                patches[i] = raw
                if mask_patches is not None:
                    mask_patches[i] = _crop(mask, cx, cy, PATCH_SIZE, fill_val=0)

        if mode == "resize56":
            # Need the CLS token -> request prefix + patch tokens.
            full = encoder.encode(patches[:m], batch_size=batch_size,
                                  include_prefix=True)  # (m, n_prefix+n_patch, 768)
            output[start:end] = full[:, 0, :]            # CLS at index 0
        elif mode == "gaussian_input":
            tokens = encoder.encode(patches[:m], batch_size=batch_size)  # (m, 256, 768)
            output[start:end] = tokens.mean(axis=1)
        else:  # mask_token
            tokens = encoder.encode(patches[:m], batch_size=batch_size)
            for i in range(m):
                w = _mask_token_weights(mask_patches[i], int(cids[start + i]),
                                        grid, encoder.patch_px)
                if w is None:
                    w = np.full(grid * grid, 1.0 / (grid * grid), dtype=np.float32)
                    n_fallback += 1
                output[start + i] = (tokens[i] * w[:, None]).sum(axis=0)

    if mode == "mask_token" and n_fallback:
        print(f"[aggregate] {n_fallback} cells had no mask coverage; "
              f"fell back to uniform pooling.")
    return output


# ----------------------------- Public API: transcriptomics -----------------------------

def preprocess_transcriptomics(
    adata: ad.AnnData,
    checkpoint: str = "MICS-Lab/novae-human-0",
    batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """Encode spatial transcriptomics through the frozen NOVAE foundation model.

    Args:
        adata: AnnData with raw counts in ``.X`` and 2D coordinates in
            ``obsm['spatial']``. NOVAE handles its own normalisation; do not
            log-transform or HVG-filter upstream.
        checkpoint: NOVAE checkpoint id (default `MICS-Lab/novae-human-0`).
        batch_size: NOVAE batch size.
        num_workers: DataLoader workers (raise above 0 on multi-CPU machines).
        device: ``cuda`` / ``cuda:0`` / ``cpu`` (defaults to cuda when available).

    Returns:
        ``(N, 64)`` ``np.float32`` NOVAE latent.
    """
    try:
        import novae
        from novae.data import quantile_scaling
    except ImportError as e:
        raise SystemExit(
            "NOVAE not installed. Run: `pip install novae`."
        ) from e

    if "spatial" not in adata.obsm:
        raise ValueError(
            "AnnData obsm['spatial'] is missing. NOVAE requires 2D coordinates."
        )
    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"obsm['spatial'] must be (N, 2); got {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError("obsm['spatial'] contains non-finite values.")
    if adata.shape[0] == 0:
        raise ValueError("Input AnnData has zero cells.")

    working = adata.copy()
    novae.spatial_neighbors(working, coord_type="generic", radius=SPATIAL_RADIUS,
                            delaunay=True, n_neighs=None)
    quantile_scaling(working)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    accelerator = "cuda" if dev.type == "cuda" and torch.cuda.is_available() else "cpu"
    model = novae.Novae.from_pretrained(checkpoint, batch_size=int(batch_size)).to(dev)
    model.eval()
    with torch.no_grad():
        model.compute_representations(working, accelerator=accelerator, zero_shot=True,
                                      num_workers=max(0, int(num_workers)))
    emb = working.obsm.get("novae_latent")
    if emb is None:
        raise RuntimeError("NOVAE did not write 'novae_latent' to obsm.")
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    return emb.astype(np.float32, copy=False)


# ----------------------------- Public API: proteomics -----------------------------

def preprocess_proteomics(
    adata: ad.AnnData,
    section_key: Optional[str] = None,
) -> np.ndarray:
    """Channel-wise arcsinh + percentile-clip + z-score for CODEX / IMC panels.

    Args:
        adata: AnnData with raw channel intensities in ``.X``.
        section_key: optional ``.obs`` column; when set, normalisation is
            applied independently within each section so slide-level
            intensity drift does not leak across sections.

    Returns:
        ``(N, D_pro)`` ``np.float32`` array of normalised protein intensities.
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    if X.size == 0:
        raise ValueError("Input adata has zero values in .X.")
    if not np.isfinite(X).all():
        n_bad = int(np.logical_not(np.isfinite(X)).sum())
        raise ValueError(f"adata.X contains {n_bad} non-finite values.")

    def _normalise(x: np.ndarray) -> np.ndarray:
        x = np.arcsinh(x / COFACTOR)
        lo = np.percentile(x, LOW_PERCENTILE, axis=0)
        hi = np.percentile(x, HIGH_PERCENTILE, axis=0)
        x = np.clip(x, lo, hi)
        return ((x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)).astype(np.float32)

    if section_key is None:
        return _normalise(X)

    if section_key not in adata.obs.columns:
        raise ValueError(
            f"section_key '{section_key}' missing from obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    sections = adata.obs[section_key].values
    for sec in np.unique(sections):
        m = sections == sec
        if int(m.sum()) < 2:
            print(f"[warn] section '{sec}' has only {int(m.sum())} cell(s); "
                  f"z-score undefined, setting to 0.")
            X[m] = 0.0
            continue
        X[m] = _normalise(X[m])
    return X
