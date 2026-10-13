#!/usr/bin/env python3
"""
Latent DRAEM: synthesize anomalies and remove them inside a *frozen* RAE latent space.

Pipeline per (encoder, synthesis mode, category):
  z    = E(x)                 frozen RAE encoder, normalized tokens (B, C, h, w), h = w = 16
  z_a  = synth(z or x)        anomaly synthesized in token space (or in pixels, then encoded)
  r    = R(z_a)               token restorer (transformer), trained to output clean z
  m    = D(z_a, r)            DRAEM-style discriminator -> token-level anomaly logits
Test: anomaly maps from D ("disc") and from cos(z, R(z)) ("resid"); no pixel decoder involved.
Baseline on the same frozen tokens: kNN distance to the train token bank ("knn", PatchCore-lite).

Synthesis modes (no real defects are ever used):
  pixel        DRAEM in pixel space (DTD texture in Perlin mask), then encoded      [reference]
  latent_tex   DTD texture *tokens* blended into a Perlin token mask
  latent_self  tokens from another normal image of the category, spatially shifted (latent CutPaste)
  latent_noise Gaussian noise on the masked tokens (latent-space variance inflation)
  mix          one of the above, chosen per sample
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
PILOT_CATEGORIES = ["carpet", "grid", "hazelnut", "screw", "transistor"]
ALL_MODES = ["pixel", "latent_tex", "latent_self", "latent_noise", "mix"]


# ============================================================================ utils
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def pushd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def list_images(d: Path, recursive=False):
    if not d.is_dir():
        return []
    it = d.rglob("*") if recursive else d.iterdir()
    return sorted(p for p in it if p.suffix.lower() in IMG_EXTS)


def load_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_mask(path: Path | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=bool)
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127


# ============================================================================ encoders
class RAEEncoder:
    """Frozen RAE loaded exactly like the official src/stage1_sample.py."""

    def __init__(self, rae_root: Path, config: Path, device):
        rae_root = rae_root.resolve()
        if str(rae_root / "src") not in sys.path:
            sys.path.insert(0, str(rae_root / "src"))
        from utils.model_utils import instantiate_from_config  # official repo
        from utils.train_utils import parse_configs
        config = config if config.is_absolute() else rae_root / config
        with pushd(rae_root):
            cfg, *_ = parse_configs(str(config))
            self.model = instantiate_from_config(cfg).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.name = config.stem
        self.device = device

    @torch.no_grad()
    def encode(self, x):  # x in [0,1], (B,3,H,W) -> (B,C,h,w) normalized latent
        z = self.model.encode(x)
        if z.dim() == 3:  # (B,N,C) if reshape_to_2d=False
            b, n, c = z.shape
            s = int(math.isqrt(n))
            z = z.transpose(1, 2).reshape(b, c, s, s)
        return z.float()

    @torch.no_grad()
    def decode(self, z):
        return self.model.decode(z).float().clamp(0, 1)


class MockEncoder:
    """Pipeline test without weights: fixed random patch projection (16x16 patches -> 768-d)."""

    def __init__(self, device, dim=768, patch=16):
        g = torch.Generator().manual_seed(0)
        self.w = (torch.randn(dim, 3 * patch * patch, generator=g) / math.sqrt(3 * patch * patch)).to(device)
        self.patch, self.dim, self.name = patch, dim, "MOCK"

    @torch.no_grad()
    def encode(self, x):
        b = x.shape[0]
        p = F.unfold(x, self.patch, stride=self.patch)  # (B, 3p^2, N)
        z = torch.einsum("dk,bkn->bdn", self.w, p - 0.5)
        s = int(math.isqrt(z.shape[-1]))
        return z.reshape(b, self.dim, s, s) * 4

    @torch.no_grad()
    def decode(self, z):
        g = z.mean(1, keepdim=True)
        return torch.sigmoid(F.interpolate(g, scale_factor=self.patch, mode="nearest")).repeat(1, 3, 1, 1)


# ============================================================================ data
def mvtec_split(root: Path, cat: str):
    train = list_images(root / cat / "train" / "good")
    test = []
    tdir = root / cat / "test"
    for d in sorted(p for p in tdir.iterdir() if p.is_dir()) if tdir.is_dir() else []:
        for p in list_images(d):
            m = None
            if d.name != "good":
                c = root / cat / "ground_truth" / d.name / f"{p.stem}_mask.png"
                m = c if c.exists() else None
            test.append((p, d.name, m))
    return train, test


def get_dtd_images(dtd_root: Path) -> list[Path]:
    imgs = list_images(dtd_root, recursive=True)
    if imgs:
        return imgs
    from torchvision.datasets import DTD  # downloads ~600 MB once
    DTD(str(dtd_root), split="train", download=True)
    return list_images(dtd_root, recursive=True)


def load_dtd_crop(path: Path, size: int, rng: random.Random) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = rng.randint(min(w, h) // 3, min(w, h))
    x0, y0 = rng.randint(0, w - s), rng.randint(0, h - s)
    img = img.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    # light colour / contrast jitter as in DRAEM's augmenters
    t = (t - t.mean()) * rng.uniform(0.6, 1.4) + t.mean() + rng.uniform(-0.15, 0.15)
    return t.clamp(0, 1)


# ============================================================================ Perlin masks
def _perlin(shape, res, rng: np.random.Generator):
    def fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.stack(np.meshgrid(np.arange(shape[0]) * res[0] / shape[0],
                                np.arange(shape[1]) * res[1] / shape[1], indexing="ij"), -1) % 1
    ang = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    grads = np.stack((np.cos(ang), np.sin(ang)), -1)

    def tile(a, b):
        return np.repeat(np.repeat(grads[a[0]:a[1], b[0]:b[1]], d[0], 0), d[1], 1)

    def dot(g, sx, sy):
        return (np.stack((grid[..., 0] + sx, grid[..., 1] + sy), -1) * g).sum(-1)

    n00 = dot(tile((0, -1), (0, -1)), 0, 0)
    n10 = dot(tile((1, None), (0, -1)), -1, 0)
    n01 = dot(tile((0, -1), (1, None)), 0, -1)
    n11 = dot(tile((1, None), (1, None)), -1, -1)
    t = fade(grid)
    n0 = n00 * (1 - t[..., 0]) + t[..., 0] * n10
    n1 = n01 * (1 - t[..., 0]) + t[..., 0] * n11
    return math.sqrt(2) * ((1 - t[..., 1]) * n0 + t[..., 1] * n1)


def perlin_mask(size: int, rng: np.random.Generator, thr=0.5, min_frac=0.003, max_frac=0.4) -> np.ndarray:
    """Binary blob mask at pixel resolution, DRAEM-style (scales 2^0..2^5 per axis)."""
    for _ in range(50):
        sx, sy = 2 ** rng.integers(0, 6), 2 ** rng.integers(0, 6)
        m = _perlin((size, size), (int(sx), int(sy)), rng) > thr
        if min_frac <= m.mean() <= max_frac:
            return m
    m = np.zeros((size, size), bool)
    c = size // 4
    y, x = rng.integers(0, size - c, 2)
    m[y:y + c, x:x + c] = True
    return m


# ============================================================================ synthesis
class Synthesizer:
    """Produces (z_anom, z_clean, token_target) batches. token_target in [0,1], (B,1,h,w)."""

    def __init__(self, enc, train_imgs: torch.Tensor, train_z: torch.Tensor, dtd_paths, args, device):
        self.enc, self.imgs, self.z = enc, train_imgs, train_z  # imgs on CPU, z on device
        self.dtd, self.args, self.dev = dtd_paths, args, device
        self.size = args.size
        self.rng = np.random.default_rng(args.seed)
        self.prng = random.Random(args.seed)
        self.dtd_z = None
        if dtd_paths and args.modes_need_dtd_tokens:
            self._build_dtd_bank()

    def _build_dtd_bank(self):
        n = min(self.args.dtd_bank, len(self.dtd))
        paths = self.prng.sample(self.dtd, n)
        zs = []
        for i in range(0, n, 32):
            x = torch.stack([load_dtd_crop(p, self.size, self.prng) for p in paths[i:i + 32]]).to(self.dev)
            zs.append(self.enc.encode(x))
        self.dtd_z = torch.cat(zs)

    def _masks(self, b):
        pm = np.stack([perlin_mask(self.size, self.rng) for _ in range(b)])
        pm = torch.from_numpy(pm).float().unsqueeze(1).to(self.dev)  # (B,1,H,W)
        h = self.z.shape[-1]
        tm = F.adaptive_avg_pool2d(pm, h)  # token coverage in [0,1]
        return pm, tm

    def sample(self, b: int, mode: str):
        idx = torch.from_numpy(self.rng.integers(0, len(self.z), b)).long()
        z = self.z[idx.to(self.z.device)].clone()
        pm, tm = self._masks(b)
        clean = torch.from_numpy(self.rng.random(b) < self.args.p_clean).to(self.dev)
        modes = [mode] * b if mode != "mix" else list(self.rng.choice(
            [m for m in ALL_MODES if m != "mix" and (m != "pixel" or self.dtd)], b))
        za = z.clone()
        for m in set(modes):
            sel = torch.tensor([mm == m for mm in modes], device=self.dev)
            if not sel.any():
                continue
            ii = sel.nonzero().squeeze(1)
            za[ii] = self._apply(m, z[ii], idx[ii.cpu()], pm[ii], tm[ii])
        za[clean] = z[clean]
        tm = tm.clone()
        tm[clean] = 0
        if not self.args.soft_targets:
            tm = (tm > 0.5).float()
        return za, z, tm

    def _apply(self, mode, z, idx, pm, tm):
        n = z.shape[0]
        beta = torch.from_numpy(self.rng.uniform(*self.args.beta, n)).float().to(self.dev).view(n, 1, 1, 1)
        if mode == "pixel":
            x = self.imgs[idx].to(self.dev)
            tex = torch.stack([load_dtd_crop(self.prng.choice(self.dtd), self.size, self.prng) for _ in range(n)]).to(self.dev)
            xa = x * (1 - pm) + (1 - beta) * pm * x + beta * pm * tex
            return self.enc.encode(xa)
        if mode == "latent_tex":
            src = self.dtd_z[torch.from_numpy(self.rng.integers(0, len(self.dtd_z), n)).to(self.dtd_z.device)]
        elif mode == "latent_self":
            j = torch.from_numpy(self.rng.integers(0, len(self.z), n)).to(self.z.device)
            src = self.z[j].clone()
            h = src.shape[-1]
            for k in range(n):  # random shift (+ optional flip) so content is misplaced
                src[k] = torch.roll(src[k], shifts=(int(self.rng.integers(2, h - 2)), int(self.rng.integers(2, h - 2))), dims=(1, 2))
                if self.rng.random() < 0.5:
                    src[k] = src[k].flip(-1)
        elif mode == "latent_noise":
            sigma = torch.from_numpy(self.rng.uniform(*self.args.noise_sigma, n)).float().to(self.dev).view(n, 1, 1, 1)
            return z + tm * sigma * torch.randn_like(z)
        else:
            raise ValueError(mode)
        return z + beta * tm * (src - z)


# ============================================================================ models
class Restorer(nn.Module):
    """Token transformer: anomalous tokens -> clean tokens (residual, zero-init head)."""

    def __init__(self, c_in=768, d=512, depth=8, heads=8, grid=16, drop_tokens=0.0):
        super().__init__()
        self.inp = nn.Linear(c_in, d)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, c_in)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.drop_tokens = drop_tokens

    def forward(self, z):
        b, c, h, w = z.shape
        t = z.flatten(2).transpose(1, 2)  # (B,N,C)
        x = self.inp(t)
        if self.training and self.drop_tokens > 0:  # optional Dinomaly-style noisy bottleneck
            x = F.dropout(x, self.drop_tokens)
        x = self.blocks(x + self.pos[:, : h * w])
        r = t + self.out(self.norm(x))
        return r.transpose(1, 2).reshape(b, c, h, w)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1),
                                 nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class Discriminator(nn.Module):
    """Input [z_a, r, z_a - r, cosdist] on the token grid -> anomaly logit per token."""

    def __init__(self, c_in=768, d=256, blocks=4):
        super().__init__()
        self.inp = nn.Conv2d(3 * c_in + 1, d, 1)
        self.blocks = nn.Sequential(*[ResBlock(d) for _ in range(blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, d), nn.GELU(), nn.Conv2d(d, 1, 1))

    def forward(self, za, r):
        cd = 1 - F.cosine_similarity(za, r, dim=1, eps=1e-6).unsqueeze(1)
        return self.out(self.blocks(self.inp(torch.cat([za, r, za - r, cd], 1))))


def focal_bce(logits, target, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    return ((1 - pt) ** gamma * ce).mean()


def restore_loss(r, z, tm, w_anom):
    cos = 1 - F.cosine_similarity(r, z, dim=1, eps=1e-6).unsqueeze(1)
    mse = (r - z).pow(2).mean(1, keepdim=True)
    w = 1 + (w_anom - 1) * tm  # up-weight anomalous tokens
    return ((cos + mse) * w).sum() / w.sum()


# ============================================================================ metrics
def to_pixel_map(tok_map: torch.Tensor, size: int, sigma: float) -> np.ndarray:
    m = F.interpolate(tok_map, size=(size, size), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    return np.stack([gaussian_filter(a, sigma) for a in m]) if sigma > 0 else m


def aupro(masks: np.ndarray, amaps: np.ndarray, max_fpr=0.3, steps=200) -> float:
    from skimage.measure import label
    if not masks.any():
        return float("nan")
    regions = [(i, label(m)) for i, m in enumerate(masks) if m.any()]
    ths = np.linspace(amaps.min(), amaps.max(), steps)[::-1]
    normal = ~masks
    n_norm = normal.sum()
    pros, fprs = [], []
    for t in ths:
        b = amaps >= t
        fprs.append((b & normal).sum() / max(n_norm, 1))
        ov = []
        for i, lab in regions:
            for r in range(1, lab.max() + 1):
                reg = lab == r
                ov.append((b[i] & reg).sum() / reg.sum())
        pros.append(np.mean(ov))
        if fprs[-1] > max_fpr:
            break
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= max_fpr
    if keep.sum() < 2:
        return float("nan")
    f, p = fprs[keep], pros[keep]
    # extend to exactly max_fpr by linear interpolation with the next point if available
    if (~keep).any():
        j = np.argmax(~keep)
        f1, p1, f2, p2 = fprs[j - 1], pros[j - 1], fprs[j], pros[j]
        p_at = p1 + (p2 - p1) * (max_fpr - f1) / max(f2 - f1, 1e-12)
        f, p = np.append(f, max_fpr), np.append(p, p_at)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(p, f) / max_fpr)


def evaluate_maps(maps: np.ndarray, masks: np.ndarray, labels: np.ndarray, do_pro: bool):
    img_scores = maps.reshape(len(maps), -1).max(1)
    out = dict(img_auroc=roc_auc_score(labels, img_scores) if labels.min() != labels.max() else float("nan"))
    flat_m, flat_s = masks.ravel(), maps.ravel()
    out["px_auroc"] = roc_auc_score(flat_m, flat_s) if flat_m.any() else float("nan")
    out["px_ap"] = average_precision_score(flat_m, flat_s) if flat_m.any() else float("nan")
    out["aupro"] = aupro(masks, maps) if do_pro else float("nan")
    return out


# ============================================================================ main routines
def encode_paths(enc, paths, size, bs=32):
    imgs, zs = [], []
    for i in range(0, len(paths), bs):
        x = torch.stack([load_rgb(p, size) for p in paths[i:i + bs]])
        imgs.append(x)
        zs.append(enc.encode(x.to(enc.device if hasattr(enc, "device") else x.device)))
    return torch.cat(imgs), torch.cat(zs)


@torch.no_grad()
def knn_maps(train_z, test_z, k=1, chunk=4096):
    dt = torch.float16 if train_z.is_cuda else torch.float32
    bank = F.normalize(train_z.flatten(2).transpose(1, 2).reshape(-1, train_z.shape[1]), dim=1).to(dt)
    out = []
    for z in test_z.split(16):
        q = F.normalize(z.flatten(2).transpose(1, 2).reshape(-1, z.shape[1]), dim=1).to(dt)
        best = []
        for qc in q.split(chunk):
            sim = qc @ bank.T
            best.append(1 - sim.topk(k, dim=1).values.float().mean(1))
        h = z.shape[-1]
        out.append(torch.cat(best).view(z.shape[0], 1, h, h))
    return torch.cat(out)


def train_one(enc, synth, mode, args, device, log_prefix=""):
    c, h = synth.z.shape[1], synth.z.shape[-1]
    R = Restorer(c, args.d_model, args.depth, args.heads, h, args.drop_tokens).to(device)
    D = Discriminator(c, args.d_disc, args.disc_blocks).to(device)
    opt = torch.optim.AdamW([{"params": R.parameters()}, {"params": D.parameters()}], lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    R.train(), D.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        za, z, tm = synth.sample(args.batch_size, mode)
        with torch.autocast(**amp):
            r = R(za)
            lr_ = restore_loss(r.float(), z, tm, args.w_anom)
            logits = D(za, r.detach() if args.detach_disc else r)
            ld = focal_bce(logits.float(), tm)
            loss = lr_ + args.w_disc * ld
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(R.parameters()) + list(D.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            print(f"{log_prefix} step {step:5d}/{args.steps}  L_rest {lr_.item():.4f}  L_disc {ld.item():.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    R.eval(), D.eval()
    return R, D


@torch.no_grad()
def infer(R, D, test_z, device, bs=32):
    disc, resid, rs = [], [], []
    for z in test_z.split(bs):
        r = R(z)
        disc.append(torch.sigmoid(D(z, r)))
        resid.append((1 - F.cosine_similarity(z, r, dim=1, eps=1e-6)).unsqueeze(1))
        rs.append(r)
    return torch.cat(disc), torch.cat(resid), torch.cat(rs)


def save_viz(enc, test_imgs, test_z, rest_z, disc_map, masks, labels, path, n=6):
    idx = [i for i in range(len(labels)) if labels[i] == 1][:n] + [i for i in range(len(labels)) if labels[i] == 0][:2]
    if not idx:
        return
    try:
        rec = enc.decode(test_z[idx])
        res = enc.decode(rest_z[idx])
    except Exception as e:  # decoder optional for viz
        print("  viz decode failed:", e)
        return
    s = test_imgs.shape[-1]
    rec = F.interpolate(rec, size=(s, s), mode="bilinear", align_corners=False).cpu()
    res = F.interpolate(res, size=(s, s), mode="bilinear", align_corners=False).cpu()
    import matplotlib.cm as cm
    rows = []
    for k, i in enumerate(idx):
        heat = torch.from_numpy(cm.inferno(np.clip(disc_map[i], 0, 1))[..., :3]).permute(2, 0, 1).float()
        gt = torch.from_numpy(masks[i]).float().unsqueeze(0).repeat(3, 1, 1)
        rows.append(torch.cat([test_imgs[i], rec[k], res[k], heat, gt], 2))
    grid = torch.cat(rows, 1).permute(1, 2, 0).numpy()
    Image.fromarray((grid.clip(0, 1) * 255).astype(np.uint8)).save(path)


def write_report(df: pd.DataFrame, out: Path):
    lines = ["# Latent DRAEM pilot", "",
             "Mean over categories. `disc` = learned discriminator map, `resid` = cos(z, R(z)), "
             "`knn` = frozen-token kNN baseline (no training).", ""]
    metrics = ["img_auroc", "px_auroc", "px_ap", "aupro"]
    for enc_name, g in df.groupby("encoder"):
        piv = g.groupby(["mode", "score"])[metrics].mean().round(3)
        try:
            tbl = piv.to_markdown()
        except ImportError:
            tbl = "```\n" + piv.to_string() + "\n```"
        lines += [f"## {enc_name}", "", tbl, ""]
    for metric in ["img_auroc", "px_auroc", "aupro"]:
        sub = df[df.score.isin(["disc", "knn"])]
        piv = sub.pivot_table(index="category", columns=["encoder", "mode"], values=metric).round(3)
        try:
            tbl = piv.to_markdown()
        except ImportError:
            tbl = "```\n" + piv.to_string() + "\n```"
        lines += [f"## {metric} per category (disc, and knn baseline)", "", tbl, ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rae-root", type=Path)
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--mock", action="store_true", help="random patch encoder, for pipeline tests")
    ap.add_argument("--mvtec-root", type=Path, default=Path(os.environ.get("MVTEC_ROOT", "/home/maometus/Documents/datasets/mvtec_anomaly_detection")))
    ap.add_argument("--dtd-root", type=Path, default=Path("data/dtd"))
    ap.add_argument("--out", type=Path, default=Path("results_latent_draem"))
    ap.add_argument("--categories", nargs="*", default=PILOT_CATEGORIES)
    ap.add_argument("--modes", nargs="*", default=ALL_MODES, choices=ALL_MODES)
    ap.add_argument("--size", type=int, default=256)
    # synthesis
    ap.add_argument("--p-clean", type=float, default=0.2)
    ap.add_argument("--beta", type=float, nargs=2, default=(0.5, 1.0))
    ap.add_argument("--noise-sigma", type=float, nargs=2, default=(0.5, 1.5))
    ap.add_argument("--dtd-bank", type=int, default=1000)
    ap.add_argument("--soft-targets", action="store_true", default=True)
    ap.add_argument("--hard-targets", dest="soft_targets", action="store_false")
    # model / training
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--drop-tokens", type=float, default=0.0)
    ap.add_argument("--d-disc", type=int, default=256)
    ap.add_argument("--disc-blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w-anom", type=float, default=5.0)
    ap.add_argument("--w-disc", type=float, default=1.0)
    ap.add_argument("--detach-disc", action="store_true", default=True)
    ap.add_argument("--joint-disc", dest="detach_disc", action="store_false")
    # eval
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--no-pro", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    args.modes_need_dtd_tokens = any(m in ("latent_tex", "mix") for m in args.modes)

    dtd = []
    if any(m in ("pixel", "latent_tex", "mix") for m in args.modes):
        dtd = get_dtd_images(args.dtd_root)
        print(f"DTD images: {len(dtd)}")
        if not dtd:
            sys.exit("No DTD images found; pass --dtd-root or drop the pixel/latent_tex/mix modes")

    encoders = []
    if args.mock:
        encoders.append(lambda: MockEncoder(device))
    for c in args.configs:
        encoders.append(lambda c=c: RAEEncoder(args.rae_root, Path(c), device))
    if not encoders:
        ap.error("pass --configs (with --rae-root) or --mock")

    res_path = args.out / "results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    done = {(r["encoder"], r["mode"], r["category"], r["score"]) for r in rows}

    for build in encoders:
        enc = build()
        enc.device = device
        print(f"\n######## encoder {enc.name} ########")
        for cat in args.categories:
            train_p, test = mvtec_split(args.mvtec_root, cat)
            if not train_p:
                print(f"[skip] {cat}: no data")
                continue
            seed_all(args.seed)
            train_imgs, train_z = encode_paths(enc, train_p, args.size)
            test_imgs, test_z = encode_paths(enc, [t[0] for t in test], args.size)
            labels = np.array([int(t[1] != "good") for t in test])
            masks = np.stack([load_mask(t[2], args.size) for t in test])
            print(f"[{enc.name}/{cat}] train {len(train_p)}  test {len(test)}  latent {tuple(train_z.shape[1:])}")

            if (enc.name, "-", cat, "knn") not in done:
                km = to_pixel_map(knn_maps(train_z, test_z), args.size, args.sigma)
                m = evaluate_maps(km, masks, labels, not args.no_pro)
                rows.append(dict(encoder=enc.name, mode="-", category=cat, score="knn", **m))
                print(f"  knn   I-AUROC {m['img_auroc']:.3f}  P-AUROC {m['px_auroc']:.3f}  AUPRO {m['aupro']:.3f}")

            synth = Synthesizer(enc, train_imgs, train_z, dtd, args, device)
            for mode in args.modes:
                if (enc.name, mode, cat, "disc") in done:
                    continue
                seed_all(args.seed)
                t0 = time.time()
                R, D = train_one(enc, synth, mode, args, device, log_prefix=f"  [{mode}]")
                dmap, rmap, rz = infer(R, D, test_z, device)
                for score, tok in (("disc", dmap), ("resid", rmap)):
                    pm = to_pixel_map(tok, args.size, args.sigma)
                    m = evaluate_maps(pm, masks, labels, not args.no_pro)
                    rows.append(dict(encoder=enc.name, mode=mode, category=cat, score=score,
                                     train_s=round(time.time() - t0), **m))
                    print(f"  {mode:12s} {score:5s} I-AUROC {m['img_auroc']:.3f}  P-AUROC {m['px_auroc']:.3f}  "
                          f"AP {m['px_ap']:.3f}  AUPRO {m['aupro']:.3f}")
                    if score == "disc" and not args.no_viz:
                        vdir = args.out / enc.name / "viz"
                        vdir.mkdir(parents=True, exist_ok=True)
                        save_viz(enc, test_imgs, test_z, rz, pm, masks, labels, vdir / f"{cat}_{mode}.png")
                pd.DataFrame(rows).to_csv(res_path, index=False)  # incremental: safe to resume
                del R, D
            del synth
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(res_path, index=False)

    write_report(pd.DataFrame(rows), args.out)
    print(f"\nDone. {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
