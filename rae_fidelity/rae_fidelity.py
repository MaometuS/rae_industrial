#!/usr/bin/env python3
"""
RAE reconstruction-fidelity pilot on MVTec AD.

Question: can a frozen RAE (frozen representation encoder + pretrained ViT pixel decoder)
reproduce normal industrial images faithfully enough to serve as the backbone of a
generate-anomaly / remove-anomaly pipeline?

For every RAE config (and optionally an SD-VAE baseline) and every MVTec category it reports:
  * Fidelity on normal images (train/good subset + test/good): PSNR, SSIM, error-map percentiles.
    -> high error here = false positives everywhere later.
  * Behaviour on defective test images:
      - PSNR inside vs outside the defect mask (does the decoder reproduce defects?)
      - pass-through ratio = mean error inside mask / mean error outside mask
        (~1 means defects are reconstructed as faithfully as normal texture)
      - pixel / image AUROC of the raw reconstruction error (expected to be low:
        a plain RAE is not trained to remove anomalies; this is a sanity number).
  * Visual grids: input | reconstruction | error map | GT mask.

Usage (on the cluster, from anywhere):
  python rae_fidelity.py --rae-root /path/to/RAE \
      --configs configs/stage1/pretrained/*.yaml \
      --mvtec-root /home/maometus/Documents/datasets/mvtec_anomaly_detection --out results/ --sdvae
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_auc_score

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ----------------------------------------------------------------------------- models
class Reconstructor:
    """Common interface: recon(x) with x in [0,1], shape (B,3,H,W) -> same shape in [0,1]."""
    name: str

    def recon(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@contextlib.contextmanager
def pushd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class RAEReconstructor(Reconstructor):
    """Loads an RAE exactly the way the official repo's src/stage1_sample.py does."""

    def __init__(self, rae_root: Path, config: Path, device: torch.device):
        rae_root = rae_root.resolve()
        src = rae_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from utils.model_utils import instantiate_from_config  # noqa: E402  (official repo)
        from utils.train_utils import parse_configs  # noqa: E402

        config = config if config.is_absolute() else (rae_root / config)
        # Checkpoint / stats paths inside the YAMLs are relative to the repo root.
        with pushd(rae_root):
            rae_config, *_ = parse_configs(str(config))
            if rae_config is None:
                raise ValueError(f"{config} has no stage_1 section")
            self.model = instantiate_from_config(rae_config).to(device).eval()
        self.name = "RAE-" + config.stem
        self.device = device

    @torch.no_grad()
    def recon(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(x)
        r = self.model.decode(z)
        if r.shape[-2:] != x.shape[-2:]:
            r = F.interpolate(r, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return r.clamp(0, 1)


class SDVAEReconstructor(Reconstructor):
    """Baseline: the SD-VAE latent used by DiAD-style diffusion anomaly detectors."""

    def __init__(self, device: torch.device, repo: str = "stabilityai/sd-vae-ft-ema"):
        from diffusers import AutoencoderKL
        self.model = AutoencoderKL.from_pretrained(repo).to(device).eval()
        self.name = "SDVAE-" + repo.split("/")[-1]

    @torch.no_grad()
    def recon(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(x * 2 - 1).latent_dist.mode()
        r = self.model.decode(z).sample
        return ((r + 1) / 2).clamp(0, 1)


class MockReconstructor(Reconstructor):
    """For testing the pipeline without weights: blur + noise (reproduces defects)."""

    def __init__(self, name="MOCK-blur"):
        self.name = name

    @torch.no_grad()
    def recon(self, x):
        k = torch.ones(3, 1, 5, 5, device=x.device) / 25
        r = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k, groups=3)
        return (r + 0.01 * torch.randn_like(r)).clamp(0, 1)


# ----------------------------------------------------------------------------- data
def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.is_dir() else []


def collect_category(root: Path, cat: str, max_train_good: int, seed: int = 0):
    """Return list of dicts: path, split, defect_type, mask_path (or None)."""
    items = []
    train_good = list_images(root / cat / "train" / "good")
    if max_train_good and len(train_good) > max_train_good:
        rng = np.random.default_rng(seed)
        train_good = [train_good[i] for i in sorted(rng.choice(len(train_good), max_train_good, replace=False))]
    items += [dict(path=p, split="train", defect="good", mask=None) for p in train_good]
    test_dir = root / cat / "test"
    for dtype_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()) if test_dir.is_dir() else []:
        for p in list_images(dtype_dir):
            mask = None
            if dtype_dir.name != "good":
                cand = root / cat / "ground_truth" / dtype_dir.name / f"{p.stem}_mask.png"
                mask = cand if cand.exists() else None
            items.append(dict(path=p, split="test", defect=dtype_dir.name, mask=mask))
    return items


def load_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=bool)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(m) > 127


# ----------------------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray, sel: np.ndarray | None = None) -> float:
    d = (a - b) ** 2
    if sel is not None:
        if sel.sum() == 0:
            return float("nan")
        d = d[sel]
    mse = float(d.mean())
    return 99.0 if mse <= 1e-10 else 10 * math.log10(1.0 / mse)


def error_map(x: np.ndarray, r: np.ndarray, sigma: float) -> np.ndarray:
    e = np.abs(x - r).mean(axis=2)
    return gaussian_filter(e, sigma=sigma) if sigma > 0 else e


def ssim_map(x: np.ndarray, r: np.ndarray):
    s, smap = structural_similarity(x, r, channel_axis=2, data_range=1.0, full=True)
    return float(s), smap.mean(axis=2)


def safe_auroc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


# ----------------------------------------------------------------------------- viz
def to_heat(e: np.ndarray, vmax: float) -> np.ndarray:
    import matplotlib.cm as cm
    return cm.inferno(np.clip(e / max(vmax, 1e-8), 0, 1))[..., :3].astype(np.float32)


def save_grid(rows, path: Path, vmax: float):
    """rows: list of (x, r, e, mask). Columns: input | recon | |x-r| | mask."""
    tiles = []
    for x, r, e, m in rows:
        mask_rgb = np.repeat(m[..., None].astype(np.float32), 3, axis=2)
        tiles.append(np.concatenate([x, r, to_heat(e, vmax), mask_rgb], axis=1))
    grid = np.concatenate(tiles, axis=0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


# ----------------------------------------------------------------------------- main loop
def evaluate(model: Reconstructor, args, device, out_dir: Path):
    per_image, per_cat = [], []
    model_dir = out_dir / model.name
    (model_dir / "grids").mkdir(parents=True, exist_ok=True)
    for cat in args.categories:
        items = collect_category(args.mvtec_root, cat, args.max_train_good)
        if not items:
            print(f"  [skip] {cat}: no images found under {args.mvtec_root / cat}")
            continue
        t0 = time.time()
        pix_scores, pix_labels = [], []
        img_scores, img_labels = [], []
        normal_err_vals = []
        grid_normal, grid_anom = [], []
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            xs = np.stack([load_rgb(it["path"], args.size) for it in batch])
            xt = torch.from_numpy(xs).permute(0, 3, 1, 2).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                rt = model.recon(xt)
            rs = rt.float().permute(0, 2, 3, 1).cpu().numpy()
            for it, x, r in zip(batch, xs, rs):
                m = load_mask(it["mask"], args.size)
                e = error_map(x, r, args.sigma)
                s, smap = ssim_map(x, r)
                is_anom = it["defect"] != "good"
                row = dict(model=model.name, category=cat, split=it["split"], defect=it["defect"],
                           file=str(it["path"].relative_to(args.mvtec_root)),
                           psnr=psnr(x, r), ssim=s,
                           err_mean=float(e.mean()), err_p99=float(np.percentile(e, 99)),
                           err_max=float(e.max()), mask_frac=float(m.mean()))
                if is_anom and m.any():
                    sel3 = np.repeat(m[..., None], 3, axis=2)
                    row.update(psnr_in_mask=psnr(x, r, sel3), psnr_out_mask=psnr(x, r, ~sel3),
                               err_in_mask=float(e[m].mean()), err_out_mask=float(e[~m].mean()),
                               ssim_in_mask=float(smap[m].mean()))
                    row["passthrough_ratio"] = row["err_in_mask"] / max(row["err_out_mask"], 1e-8)
                per_image.append(row)
                if it["split"] == "test":
                    img_scores.append(float(e.max()))
                    img_labels.append(int(is_anom))
                    # subsample pixels to keep memory bounded
                    stride = args.pixel_stride
                    pix_scores.append(e[::stride, ::stride].ravel())
                    pix_labels.append(m[::stride, ::stride].ravel())
                if not is_anom:
                    normal_err_vals.append(e[::4, ::4].ravel())
                    if len(grid_normal) < args.grid_n:
                        grid_normal.append((x, r, e, m))
                elif len(grid_anom) < args.grid_n:
                    grid_anom.append((x, r, e, m))

        df = pd.DataFrame([r for r in per_image if r["category"] == cat and r["model"] == model.name])
        normal = df[df.defect == "good"]
        anom = df[df.defect != "good"]
        nev = np.concatenate(normal_err_vals) if normal_err_vals else np.array([np.nan])
        ps = np.concatenate(pix_scores) if pix_scores else np.array([])
        pl = np.concatenate(pix_labels) if pix_labels else np.array([], dtype=bool)
        above = float(np.mean(ps[pl] > np.nanpercentile(nev, 99))) if pl.any() else float("nan")
        summary = dict(
            model=model.name, category=cat,
            n_normal=len(normal), n_anom=len(anom),
            normal_psnr=normal.psnr.mean(), normal_ssim=normal.ssim.mean(),
            normal_err_p95=float(np.nanpercentile(nev, 95)), normal_err_p99=float(np.nanpercentile(nev, 99)),
            normal_img_err_max_mean=normal.err_max.mean(),
            anom_psnr_in_mask=anom.get("psnr_in_mask", pd.Series(dtype=float)).mean(),
            anom_psnr_out_mask=anom.get("psnr_out_mask", pd.Series(dtype=float)).mean(),
            passthrough_ratio=anom.get("passthrough_ratio", pd.Series(dtype=float)).median(),
            # fraction of defect pixels whose error exceeds the normal-image p99 error
            defect_px_above_normal_p99=above,
            pixel_auroc_raw=safe_auroc(pl, ps) if len(ps) else float("nan"),
            image_auroc_raw=safe_auroc(img_labels, img_scores),
            seconds=time.time() - t0,
        )
        per_cat.append(summary)
        vmax = summary["normal_err_p99"] * 3 if np.isfinite(summary["normal_err_p99"]) else 0.2
        if grid_normal:
            save_grid(grid_normal, model_dir / "grids" / f"{cat}_normal.png", vmax)
        if grid_anom:
            save_grid(grid_anom, model_dir / "grids" / f"{cat}_defect.png", vmax)
        print(f"  {cat:12s} normal PSNR {summary['normal_psnr']:.2f}  SSIM {summary['normal_ssim']:.3f}  "
              f"pass-through {summary['passthrough_ratio']:.2f}  pixAUROC(raw) {summary['pixel_auroc_raw']:.3f}  "
              f"({summary['seconds']:.0f}s)")
    return per_image, per_cat


def md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string() + "\n```"


def write_report(cat_df: pd.DataFrame, out_dir: Path):
    cols = ["normal_psnr", "normal_ssim", "normal_err_p99", "anom_psnr_in_mask", "passthrough_ratio",
            "defect_px_above_normal_p99", "pixel_auroc_raw", "image_auroc_raw"]
    lines = ["# RAE reconstruction fidelity on MVTec AD", ""]
    mean = cat_df.groupby("model")[cols].mean(numeric_only=True)
    lines += ["## Mean over categories", "", md(mean.round(3)), ""]
    for metric in ["normal_psnr", "normal_ssim", "passthrough_ratio"]:
        piv = cat_df.pivot(index="category", columns="model", values=metric)
        lines += [f"## {metric} by category", "", md(piv.round(3)), ""]
    lines += [
        "## How to read this",
        "",
        "- normal_psnr / normal_ssim: fidelity on defect-free images. Low values = the frozen decoder "
        "loses industrial detail -> false positives later. Compare RAE rows against the SD-VAE row.",
        "- normal_err_p99: the reconstruction noise floor on normal images (99th pct of smoothed |x-r|).",
        "- anom_psnr_in_mask / passthrough_ratio: how faithfully defects themselves are reproduced. "
        "Ratio ~1 = defects reconstructed as well as normal texture (expected for a plain RAE; "
        "the restorer has to learn removal).",
        "- defect_px_above_normal_p99: share of defect pixels whose raw error already exceeds the normal noise floor.",
        "- pixel/image_auroc_raw: raw reconstruction error as detector. Not the target metric; sanity check only.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rae-root", type=Path, help="clone of the official RAE repo (with models/ downloaded)")
    ap.add_argument("--configs", nargs="*", default=[], help="stage-1 RAE YAMLs (relative to --rae-root or absolute)")
    ap.add_argument("--mvtec-root", type=Path, default=Path(os.environ.get("MVTEC_ROOT", "/home/maometus/Documents/datasets/mvtec_anomaly_detection")))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--categories", nargs="*", default=MVTEC_CATEGORIES)
    ap.add_argument("--size", type=int, default=256, help="input resolution fed to the autoencoder")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-train-good", type=int, default=50, help="train/good images per category (0 = all)")
    ap.add_argument("--sigma", type=float, default=4.0, help="gaussian smoothing of error maps (0 = off)")
    ap.add_argument("--pixel-stride", type=int, default=2, help="pixel subsampling for AUROC")
    ap.add_argument("--grid-n", type=int, default=6, help="examples per visual grid")
    ap.add_argument("--sdvae", action="store_true", help="also run the SD-VAE baseline (needs diffusers)")
    ap.add_argument("--mock", action="store_true", help="run a blur stub instead of real models (pipeline test)")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast on CUDA")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    builders = []
    if args.mock:
        builders.append(lambda: MockReconstructor())
    for c in args.configs:
        if args.rae_root is None:
            ap.error("--configs requires --rae-root")
        builders.append(lambda c=c: RAEReconstructor(args.rae_root, Path(c), device))
    if args.sdvae:
        builders.append(lambda: SDVAEReconstructor(device))
    if not builders:
        ap.error("nothing to run: pass --configs, --sdvae or --mock")

    all_img, all_cat = [], []
    for build in builders:
        model = build()
        print(f"\n=== {model.name} on {device} ===")
        img_rows, cat_rows = evaluate(model, args, device, args.out)
        all_img += img_rows
        all_cat += cat_rows
        # write incrementally so a crash in a later model keeps earlier results
        pd.DataFrame(all_img).to_csv(args.out / "per_image.csv", index=False)
        pd.DataFrame(all_cat).to_csv(args.out / "per_category.csv", index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_report(pd.DataFrame(all_cat), args.out)
    print(f"\nDone. See {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
