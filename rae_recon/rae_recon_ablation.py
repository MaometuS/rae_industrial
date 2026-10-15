#!/usr/bin/env python3
"""
RAE as a reconstruction-based anomaly detector: which anti-copying mechanisms make it work?

    x ─► frozen RAE encoder ─► z ─► [S] [M] [N] ─► RAE decoder (fine-tuned) ─► x̂ ≈ normal image
                                                                 score: compare x with x̂

The RAE latent (16×16×768 = 196,608 values) is as large as the image (256×256×3 = 196,608), so a plainly
fine-tuned decoder can copy defects. Three mechanisms that block copying, applied to the tokens before the
decoder during fine-tuning, and all 8 combinations of them:

  M  masking         random tokens replaced by a learned mask token; decoder paints them from context.
                     Test time: 4 complementary passes (token grid parity groups), each patch taken from the pass
                     in which it was masked.
  N  noisy bottleneck feature dropout + Gaussian noise on the tokens (Dinomaly-style; RAE's decoder was trained
                     with latent noise already).
  S  synthetic anomalies  latent / pixel anomalies (latent_draem Synthesizer, default mode "mix");
                     the decoder must output the clean image.
  none               plain decoder fine-tuning on normals: the naive baseline (expected to copy defects).

Scores per run: pix (|x − x̂|), ssim (1 − SSIM), feat (1 − cos(E(x), E(x̂)) per token), knn (frozen-token control),
plus fusions z-normalized on held-out normal images: pix+feat, feat+knn, pix+feat+knn.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "latent_draem"))
from latent_draem import (  # noqa: E402  (shared utilities from the Exp 2/3 code)
    ALL_MODES, PILOT_CATEGORIES, RAEEncoder, Synthesizer, encode_paths, evaluate_maps, get_dtd_images,
    knn_maps, load_mask, mvtec_split, seed_all, to_pixel_map,
)

COMBOS = ["none", "M", "N", "S", "MN", "MS", "NS", "MNS"]


# ============================================================================ mock model (pipeline tests only)
class _MockModel(nn.Module):
    def __init__(self, dim=768, patch=16):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("w", torch.randn(dim, 3 * patch * patch, generator=g) / math.sqrt(3 * patch * patch))
        self.decoder = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 3 * patch * patch))
        self.patch, self.dim = patch, dim

    @torch.no_grad()
    def encode(self, x):
        p = F.unfold(x, self.patch, stride=self.patch)
        z = torch.einsum("dk,bkn->bdn", self.w, p - 0.5)
        s = int(math.isqrt(z.shape[-1]))
        return (z * 4).reshape(x.shape[0], self.dim, s, s)

    def decode(self, z):
        b, c, h, w = z.shape
        t = self.decoder(z.flatten(2).transpose(1, 2)).transpose(1, 2)  # (B, 3p², N)
        return torch.sigmoid(F.fold(t, (h * self.patch, w * self.patch), self.patch, stride=self.patch))


class MockRAE:
    def __init__(self, device):
        self.model = _MockModel().to(device)
        self.name, self.device = "MOCK", device

    @torch.no_grad()
    def encode(self, x):
        return self.model.encode(x)


# ============================================================================ losses / maps
def _gauss_win(size=11, sigma=1.5, ch=3, device="cpu"):
    g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).float()
    return (g[:, None] @ g[None, :]).expand(ch, 1, size, size).contiguous()


def ssim_map(a, b, win=None):
    """Per-pixel SSIM (B,1,H,W) for images in [0,1]."""
    ch = a.shape[1]
    win = _gauss_win(ch=ch, device=a.device) if win is None else win
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=ch)
    mu_b = F.conv2d(b, win, padding=pad, groups=ch)
    s_aa = F.conv2d(a * a, win, padding=pad, groups=ch) - mu_a ** 2
    s_bb = F.conv2d(b * b, win, padding=pad, groups=ch) - mu_b ** 2
    s_ab = F.conv2d(a * b, win, padding=pad, groups=ch) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean(1, keepdim=True)


def blur(maps: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([gaussian_filter(m, sigma) for m in maps]) if sigma > 0 else maps


# ============================================================================ model wrapper
class Reconstructor:
    """Frozen encoder + trainable decoder + learned mask token, with the M/N/S mechanisms."""

    def __init__(self, enc, combo: str, args, device):
        self.enc, self.combo, self.args, self.dev = enc, combo, args, device
        self.M, self.N, self.S = ("M" in combo), ("N" in combo), ("S" in combo)
        c = None  # latent channels, filled on first use
        self.mask_token = None
        self._c = c

    def ensure_mask_token(self, c):
        if self.mask_token is None:
            self.mask_token = nn.Parameter(torch.zeros(1, c, 1, 1, device=self.dev))

    def decode(self, z):
        x = self.enc.model.decode(z)
        size = self.args.size
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return x

    def corrupt(self, z):
        """Training-time M and N on (already S-corrupted) tokens."""
        b, c, h, w = z.shape
        if self.M:
            self.ensure_mask_token(c)
            ratio = torch.empty(b, 1, 1, 1, device=z.device).uniform_(*self.args.mask_ratio)
            m = (torch.rand(b, 1, h, w, device=z.device) < ratio).float()
            z = z * (1 - m) + self.mask_token * m
        if self.N:
            z = F.dropout(z, p=self.args.feat_dropout, training=True)
            sig = torch.empty(b, 1, 1, 1, device=z.device).uniform_(0, self.args.noise_tau)
            z = z + sig * torch.randn_like(z)
        return z

    @torch.no_grad()
    def reconstruct(self, z):
        """Test-time reconstruction. With M: 4 complementary parity-group passes, patch-wise assembly."""
        if not self.M:
            return self.decode(z).float().clamp(0, 1)
        b, c, h, w = z.shape
        self.ensure_mask_token(c)
        ii, jj = torch.meshgrid(torch.arange(h, device=z.device), torch.arange(w, device=z.device), indexing="ij")
        groups = (ii % 2) * 2 + (jj % 2)
        out = torch.zeros(b, 3, self.args.size, self.args.size, device=z.device)
        for g in range(4):
            m = (groups == g).float()[None, None]
            xg = self.decode(z * (1 - m) + self.mask_token * m).float()
            mp = F.interpolate(m, size=out.shape[-2:], mode="nearest")
            out = out + xg * mp
        return out.clamp(0, 1)


def sample_batch(synth: Synthesizer, imgs, z_all, args, use_S: bool):
    """Like Synthesizer.sample but also returns the clean images (decoder targets)."""
    b = args.batch_size
    idx = torch.from_numpy(synth.rng.integers(0, len(z_all), b)).long()
    z = z_all[idx.to(z_all.device)]
    x = imgs[idx].to(z.device)
    if not use_S:
        return z, x
    pm, tm = synth._masks(b)
    clean = torch.from_numpy(synth.rng.random(b) < args.p_clean).to(z.device)
    mode = args.synth_mode
    modes = [mode] * b if mode != "mix" else list(synth.rng.choice([m for m in ALL_MODES if m != "mix"], b))
    za = z.clone()
    for m in set(modes):
        sel = torch.tensor([mm == m for mm in modes], device=z.device)
        ii = sel.nonzero().squeeze(1)
        za[ii] = synth._apply(m, z[ii], idx[ii.cpu()], pm[ii], tm[ii])
    za[clean] = z[clean]
    return za, x


def train_one(rec: Reconstructor, synth, imgs, train_z, args, device, tag=""):
    dec = rec.enc.model.decoder
    dec.train()
    for p in dec.parameters():
        p.requires_grad_(True)
    if rec.M:
        rec.ensure_mask_token(train_z.shape[1])
    groups = [{"params": [p for p in dec.parameters() if p.requires_grad], "lr": args.lr}]
    if rec.mask_token is not None:
        groups.append({"params": [rec.mask_token], "lr": args.lr_mask})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps, pct_start=0.05)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    win = _gauss_win(device=device)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        za, x = sample_batch(synth, imgs, train_z, args, rec.S)
        zin = rec.corrupt(za)
        with torch.autocast(**amp):
            xh = rec.decode(zin)
        xh = xh.float()
        l1 = (xh - x).abs().mean()
        ls = 1 - ssim_map(xh.clamp(0, 1), x, win).mean()
        loss = l1 + args.w_ssim * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            print(f"{tag} step {step:5d}/{args.steps}  L1 {l1.item():.4f}  1-SSIM {ls.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    dec.eval()
    return time.time() - t0


@torch.no_grad()
def score_maps(rec: Reconstructor, imgs, z, args, bs=16):
    """Returns pixel-level maps (N,H,W): pix, ssim, feat; and reconstructions."""
    win = _gauss_win(device=rec.dev)
    pix, ss, feat, recs = [], [], [], []
    for i in range(0, len(z), bs):
        zb = z[i:i + bs]
        xb = imgs[i:i + bs].to(rec.dev)
        xh = rec.reconstruct(zb)
        pix.append((xh - xb).abs().mean(1).cpu().numpy())
        ss.append((1 - ssim_map(xh, xb, win))[:, 0].cpu().numpy())
        zh = rec.enc.encode(xh)
        cd = (1 - F.cosine_similarity(zb, zh, dim=1, eps=1e-6)).unsqueeze(1)
        feat.append(F.interpolate(cd, size=xb.shape[-2:], mode="bilinear", align_corners=False)[:, 0].cpu().numpy())
        recs.append(xh.cpu())
    s = args.sigma
    return dict(pix=blur(np.concatenate(pix), s), ssim=blur(np.concatenate(ss), s),
                feat=blur(np.concatenate(feat), s)), torch.cat(recs)


def znorm(test, val):
    v = val.astype(np.float32)
    return (test.astype(np.float32) - v.mean()) / (v.std() + 1e-6)


def save_viz(imgs, recs, maps, masks, labels, path, n_anom=4, n_norm=2):
    import matplotlib.cm as cm
    idx = [i for i in range(len(labels)) if labels[i]][:n_anom] + [i for i in range(len(labels)) if not labels[i]][:n_norm]
    if not idx:
        return
    rows = []
    vmax = {k: np.percentile(maps[k], 99.5) + 1e-8 for k in ("pix", "feat")}
    for i in idx:
        heat = lambda k: torch.from_numpy(cm.inferno(np.clip(maps[k][i] / vmax[k], 0, 1))[..., :3]).permute(2, 0, 1).float()
        gt = torch.from_numpy(masks[i]).float()[None].repeat(3, 1, 1)
        rows.append(torch.cat([imgs[i], recs[i], heat("pix"), heat("feat"), gt], 2))
    grid = torch.cat(rows, 1).permute(1, 2, 0).numpy()
    Image.fromarray((grid.clip(0, 1) * 255).astype(np.uint8)).save(path)


def write_report(df: pd.DataFrame, out: Path):
    metrics = [m for m in ["img_auroc", "img_auroc_top1", "px_auroc", "px_ap", "aupro"] if m in df]
    md = lambda d: d.to_markdown() if _has_tabulate() else "```\n" + d.to_string() + "\n```"
    lines = ["# RAE reconstruction ablation (decoder fine-tuning with M / N / S)", "",
             "Mean over categories; ± = std over seeds (shown when >1 seed). `knn` is the frozen-token control "
             "and does not depend on the combination.", ""]
    multi = df[df.score != "knn"].seed.nunique() > 1
    per_seed = df.groupby(["combo", "score", "seed"])[metrics].mean()
    mu = per_seed.groupby(["combo", "score"]).mean()
    sd = per_seed.groupby(["combo", "score"]).std().fillna(0)
    tbl = mu.round(3).astype(str) + (" ± " + sd.round(3).astype(str) if multi else "")
    order = ["none", "M", "N", "S", "MN", "MS", "NS", "MNS", "-"]
    tbl = tbl.reindex(sorted(tbl.index, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1])))
    lines += ["## All combinations × scores", "", md(tbl), ""]
    for score in ["feat", "pix", "pix+feat+knn"]:
        sub = df[df.score == score]
        if sub.empty:
            continue
        piv = sub.groupby(["category", "combo"])["aupro"].mean().unstack("combo")
        piv = piv[[c for c in order if c in piv.columns]].round(3)
        k = df[df.score == "knn"].groupby("category")["aupro"].mean().round(3)
        piv.insert(0, "knn", k)
        lines += [f"## AUPRO per category: `{score}` (knn column = control)", "", md(piv), ""]
    for score in ["feat", "pix"]:
        sub = df[df.score == score].groupby(["combo", "seed"])[metrics].mean().groupby("combo").mean()
        if "none" in sub.index:
            d = (sub - sub.loc["none"]).drop(index="none").round(3)
            lines += [f"## Gain over the naive baseline (`none`), score `{score}`", "", md(d), ""]
    (out / "report.md").write_text("\n".join(lines))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rae-root", type=Path)
    ap.add_argument("--config", default="configs/stage1/pretrained/DINOv2-B.yaml")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mvtec-root", type=Path, default=Path("/home/maometus/Documents/datasets/mvtec_anomaly_detection"))
    ap.add_argument("--dtd-root", type=Path, default=HERE.parent / "latent_draem" / "data" / "dtd")
    ap.add_argument("--out", type=Path, default=HERE / "results_recon_ablation")
    ap.add_argument("--categories", nargs="*", default=PILOT_CATEGORIES)
    ap.add_argument("--combos", nargs="*", default=COMBOS, choices=COMBOS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=20, help="held-out train/good images for score normalization")
    # mechanisms
    ap.add_argument("--mask-ratio", type=float, nargs=2, default=(0.25, 0.5))
    ap.add_argument("--feat-dropout", type=float, default=0.2)
    ap.add_argument("--noise-tau", type=float, default=0.8)
    ap.add_argument("--synth-mode", default="mix", choices=ALL_MODES)
    ap.add_argument("--p-clean", type=float, default=0.2)
    ap.add_argument("--beta", type=float, nargs=2, default=(0.5, 1.0))
    ap.add_argument("--noise-sigma", type=float, nargs=2, default=(0.5, 1.5))
    ap.add_argument("--dtd-bank", type=int, default=1000)
    # training
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5, help="decoder learning rate (fine-tuning)")
    ap.add_argument("--lr-mask", type=float, default=1e-3)
    ap.add_argument("--w-ssim", type=float, default=0.5)
    # eval
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--no-pro", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    needs_S = any("S" in c for c in args.combos)
    dtd = get_dtd_images(args.dtd_root) if needs_S and args.synth_mode in ("pixel", "latent_tex", "mix") else []
    if needs_S and args.synth_mode in ("pixel", "latent_tex", "mix") and not dtd:
        sys.exit("S needs DTD images; pass --dtd-root")

    if args.mock:
        enc = MockRAE(device)
    else:
        if args.rae_root is None:
            ap.error("--rae-root is required (or --mock)")
        enc = RAEEncoder(args.rae_root, Path(args.config), device)
        enc.device = device
    pristine = copy.deepcopy(enc.model.decoder.state_dict())
    for k in pristine:
        pristine[k] = pristine[k].detach().cpu().clone()

    res_path = args.out / "results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    done = {(r["combo"], r["category"], int(r["seed"]), r["score"]) for r in rows}

    for cat in args.categories:
        train_p, test = mvtec_split(args.mvtec_root, cat)
        if not train_p:
            print(f"[skip] {cat}")
            continue
        perm = np.random.default_rng(1234).permutation(len(train_p))
        val_p = [train_p[i] for i in sorted(perm[:args.n_val])] if args.n_val else []
        train_p = [train_p[i] for i in sorted(perm[args.n_val:])] if args.n_val else train_p
        train_imgs, train_z = encode_paths(enc, train_p, args.size)
        test_imgs, test_z = encode_paths(enc, [t[0] for t in test], args.size)
        val_imgs, val_z = encode_paths(enc, val_p, args.size) if val_p else (None, None)
        labels = np.array([int(t[1] != "good") for t in test])
        masks = np.stack([load_mask(t[2], args.size) for t in test])
        print(f"\n=== {cat}: train {len(train_p)} val {len(val_p)} test {len(test)} latent {tuple(train_z.shape[1:])}")

        knn_test = to_pixel_map(knn_maps(train_z, test_z), args.size, args.sigma)
        knn_val = to_pixel_map(knn_maps(train_z, val_z), args.size, args.sigma) if val_z is not None else None
        if ("-", cat, 0, "knn") not in done:
            m = evaluate_maps(knn_test, masks, labels, not args.no_pro)
            rows.append(dict(combo="-", category=cat, seed=0, score="knn", **m))
            print(f"  knn                I {m['img_auroc']:.3f}  P {m['px_auroc']:.3f}  AP {m['px_ap']:.3f}  PRO {m['aupro']:.3f}")

        for seed in args.seeds:
            synth = None
            if needs_S:
                sargs = SimpleNamespace(size=args.size, seed=seed, p_clean=args.p_clean, beta=args.beta,
                                        noise_sigma=args.noise_sigma, dtd_bank=args.dtd_bank, soft_targets=True,
                                        modes_need_dtd_tokens=args.synth_mode in ("latent_tex", "mix"))
                synth = Synthesizer(enc, train_imgs, train_z, dtd, sargs, device)
            for combo in args.combos:
                if (combo, cat, seed, "feat") in done:
                    continue
                seed_all(seed)
                enc.model.decoder.load_state_dict(pristine)
                rec = Reconstructor(enc, combo, args, device)
                s = synth if synth is not None else SimpleNamespace(rng=np.random.default_rng(seed))
                secs = train_one(rec, s, train_imgs, train_z, args, device, tag=f"  [{combo} s{seed}]")
                maps, recs = score_maps(rec, test_imgs, test_z, args)
                maps["knn"] = knn_test
                if val_z is not None:
                    vmaps, _ = score_maps(rec, val_imgs, val_z, args)
                    vmaps["knn"] = knn_val
                    zt = {k: znorm(maps[k], vmaps[k]) for k in ("pix", "feat", "knn")}
                    maps["pix+feat"] = (zt["pix"] + zt["feat"]) / 2
                    maps["feat+knn"] = (zt["feat"] + zt["knn"]) / 2
                    maps["pix+feat+knn"] = (zt["pix"] + zt["feat"] + zt["knn"]) / 3
                for score, mp in maps.items():
                    if score == "knn":
                        continue
                    m = evaluate_maps(mp, masks, labels, not args.no_pro)
                    rows.append(dict(combo=combo, category=cat, seed=seed, score=score, train_s=round(secs), **m))
                    print(f"  {combo:4s} s{seed} {score:13s} I {m['img_auroc']:.3f}/{m['img_auroc_top1']:.3f}  "
                          f"P {m['px_auroc']:.3f}  AP {m['px_ap']:.3f}  PRO {m['aupro']:.3f}", flush=True)
                if not args.no_viz and seed == args.seeds[0]:
                    vdir = args.out / "viz"
                    vdir.mkdir(exist_ok=True)
                    save_viz(test_imgs, recs, maps, masks, labels, vdir / f"{cat}_{combo}.png")
                pd.DataFrame(rows).to_csv(res_path, index=False)
                del rec
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        enc.model.decoder.load_state_dict(pristine)

    write_report(pd.DataFrame(rows), args.out)
    print(f"\nDone. {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
