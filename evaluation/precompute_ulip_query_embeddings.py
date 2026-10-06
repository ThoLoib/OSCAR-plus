"""
precompute_ulip_query_embeddings.py
====================================

Standalone pre-encoder for ULIP-2 cross-modal query embeddings.

Run this BEFORE experiments/stage2_mi3dor.py when the GPU does not have
enough VRAM to hold ViT-bigG-14 alongside CLIP + DINOv2 + PointBERT.

Strategy
--------
Load ONLY OpenCLIP ViT-bigG-14 in float16 (~5 GB).  With nothing else in
VRAM, it fits on a 6 GB GPU.  Batch-encode all query images, save embeddings
to a .pt cache file.  The main eval script detects the cache and skips the
per-query forward pass entirely.

How to run
----------
    python3 evaluation/precompute_ulip_query_embeddings.py

Paths default to ``config/paths.yaml`` and every one can be overridden on the
command line (``--datasets-root``, ``--caches-root``, ...).  Outside the
container the script wraps itself into ``docker compose run`` automatically.
"""

import os
import sys

# ---------------------------------------------------------------------------
# CLI prelude — before the heavy imports (torch/open_clip) so that `--help`
# works on the host without the container dependencies.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_PATHS = None
_BATCH_SIZE = 8        # the reported numbers were produced with this value

if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="images per forward pass (default: 8). A different "
                             "batch size changes the float16 accumulation and "
                             "shifts the embeddings in the fifth decimal, so "
                             "keep the default to reproduce the reported "
                             "Stage-2 numbers")
    parser.add_argument("--no-docker", action="store_true",
                        help="do not auto-wrap into the oscar-plus container")
    _paths_mod.add_path_args(parser)
    args = parser.parse_args()

    _BATCH_SIZE = args.batch_size
    _PATHS = _paths_mod.from_args(args)

    if not os.path.exists("/.dockerenv") and not args.no_docker:
        print("[precompute] runs in the oscar-plus container — wrapping "
              "automatically.", flush=True)
        os.chdir(_REPO)                      # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3",
                             "/app/evaluation/precompute_ulip_query_embeddings.py"]
                            + sys.argv[1:])

if _PATHS is None:      # imported as a module (no CLI): paths.yaml defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

bop_root              = os.path.join(_PATHS["datasets_root"], "mi3dor/image/test")
ulip_query_cache_path = os.path.join(_PATHS["caches_root"],
                                     "ulip_query_cache_mi3dor.pt")  # output

# OpenCLIP model (must match what ULIP-2 uses)
openclip_model      = "ViT-bigG-14"
openclip_pretrained = "laion2b_s39b_b160k"

# Encoding settings
batch_size = _BATCH_SIZE
dtype      = torch.float16   # float16 halves VRAM vs float32
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Collect query image paths
# ============================================================================

def collect_query_paths(bop_root: str):
    if not os.path.isdir(bop_root):
        raise FileNotFoundError(f"bop_root not found: {bop_root}")
    paths = []
    for category in sorted(os.listdir(bop_root)):
        cat_dir = os.path.join(bop_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(cat_dir, fname))
    return paths


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"[precompute] device = {device}")
    print(f"[precompute] dtype  = {dtype}")
    print(f"[precompute] batch  = {batch_size}")

    # --- Load model ---
    try:
        import open_clip
    except ImportError:
        sys.exit("open_clip not installed. Run: pip install open-clip-torch")

    print(f"[precompute] Loading {openclip_model} ({openclip_pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        openclip_model, pretrained=openclip_pretrained
    )
    model = model.visual          # we only need the image tower
    model = model.to(dtype).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[precompute] Model loaded: {n_params / 1e6:.0f}M params")

    # --- Collect paths ---
    img_paths = collect_query_paths(bop_root)
    print(f"[precompute] {len(img_paths)} query images found under {bop_root}")

    # --- Preprocess (CPU, one-time) ---
    print("[precompute] Preprocessing images...")
    tensors, valid_paths = [], []
    for p in tqdm(img_paths, desc="preprocess", unit="img"):
        try:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
            valid_paths.append(p)
        except Exception as exc:
            tqdm.write(f"[precompute] skip {p}: {exc}")
    print(f"[precompute] {len(tensors)} images preprocessed.")

    # --- Encode in batches ---
    cache = {}
    for i in tqdm(range(0, len(tensors), batch_size),
                  desc="encode", unit="batch"):
        batch_tensors = tensors[i:i + batch_size]
        batch = torch.stack(batch_tensors).to(device)
        if dtype == torch.float16:
            batch = batch.half()

        try:
            with torch.no_grad():
                emb = model(batch)                      # (B, embed_dim)
                emb = F.normalize(emb.float(), p=2, dim=-1)  # store as float32
        except torch.cuda.OutOfMemoryError:
            print(f"\n[precompute] OOM at batch_size={batch_size}. "
                  "Restart with a smaller --batch-size.")
            raise

        batch_paths = valid_paths[i:i + batch_size]
        for j, p in enumerate(batch_paths):
            cache[p] = emb[j:j + 1].cpu()              # (1, embed_dim), float32

    # --- Save ---
    print(f"[precompute] Saving {len(cache)} embeddings → {ulip_query_cache_path}")
    os.makedirs(os.path.dirname(ulip_query_cache_path) or ".", exist_ok=True)
    torch.save(cache, ulip_query_cache_path)
    size_mb = os.path.getsize(ulip_query_cache_path) / 1024 / 1024
    print(f"[precompute] Done. Cache size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
