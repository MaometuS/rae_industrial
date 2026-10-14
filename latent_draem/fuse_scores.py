#!/usr/bin/env python3
"""
Offline score fusion for Experiment 3 (needs a run with --save-maps --n-val N).

For every (encoder, category, mode, seed) it loads the saved token-level maps, z-normalizes each
score with the mean/std of its maps on the held-out *normal* validation images (no test labels used),
and evaluates single scores and fusions:
    disc, resid, knn, disc+knn, resid+knn, disc+resid+knn   (fusion = mean of z-scores)
Every combination is scored with both image scores (max and top-1% mean).

Writes <out>/fusion_results.csv and <out>/report_fusion.md (mean ± std over seeds).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_draem import evaluate_maps, to_pixel_map  # noqa: E402

FUSIONS = {
    "disc": ["disc"], "resid": ["resid"], "knn": ["knn"],
    "disc+knn": ["disc", "knn"], "resid+knn": ["resid", "knn"], "disc+resid+knn": ["disc", "resid", "knn"],
}
METRICS = ["img_auroc", "img_auroc_top1", "px_auroc", "px_ap", "aupro"]


def znorm(test: np.ndarray, val: np.ndarray) -> np.ndarray:
    v = val.astype(np.float32)
    return (test.astype(np.float32) - v.mean()) / (v.std() + 1e-6)


def md(df):
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="experiment folder containing maps/")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--no-pro", action="store_true")
    args = ap.parse_args()

    root = args.out / "maps"
    if not root.is_dir():
        sys.exit(f"{root} not found: run latent_draem.py with --save-maps --n-val N first")
    res_path = args.out / "fusion_results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    done = {(r["encoder"], r["category"], r["mode"], int(r["seed"]), r["score"]) for r in rows}

    for enc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cat_dir in sorted(p for p in enc_dir.iterdir() if p.is_dir()):
            meta = np.load(cat_dir / "meta.npz")
            labels = meta["labels"]
            shape = tuple(meta["mask_shape"])
            masks = np.unpackbits(meta["masks"], axis=-1)[..., : shape[-1]].astype(bool)
            knn = np.load(cat_dir / "knn.npz")
            if knn["val"].size == 0:
                sys.exit(f"{cat_dir}: no validation maps; rerun with --n-val > 0")
            base = {"knn": znorm(knn["test"], knn["val"])}
            if (enc_dir.name, cat_dir.name, "-", 0, "knn") not in done:  # kNN has no seed: evaluate once
                pm = to_pixel_map(torch.from_numpy(base["knn"]).float(), args.size, args.sigma)
                rows.append(dict(encoder=enc_dir.name, category=cat_dir.name, mode="-", seed=0, score="knn",
                                 **evaluate_maps(pm, masks, labels, not args.no_pro)))
            for f in sorted(cat_dir.glob("*_s*.npz")):
                mode, seed = re.match(r"(.+)_s(\d+)\.npz", f.name).groups()
                seed = int(seed)
                d = np.load(f)
                maps = dict(base, disc=znorm(d["disc"], d["disc_val"]), resid=znorm(d["resid"], d["resid_val"]))
                for name, parts in FUSIONS.items():
                    if name == "knn":
                        continue
                    if (enc_dir.name, cat_dir.name, mode, seed, name) in done:
                        continue
                    fused = np.mean([maps[p] for p in parts], axis=0)
                    pm = to_pixel_map(torch.from_numpy(fused).float(), args.size, args.sigma)
                    m = evaluate_maps(pm, masks, labels, not args.no_pro)
                    rows.append(dict(encoder=enc_dir.name, category=cat_dir.name, mode=mode, seed=seed, score=name, **m))
                    print(f"{enc_dir.name}/{cat_dir.name} {mode} s{seed} {name:15s} "
                          f"I {m['img_auroc']:.3f}/{m['img_auroc_top1']:.3f}  P {m['px_auroc']:.3f}  "
                          f"AP {m['px_ap']:.3f}  PRO {m['aupro']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(res_path, index=False)

    df = pd.DataFrame(rows)
    lines = ["# Experiment 3: score fusion", "",
             "Each score z-normalized with its maps on held-out normal train images; fusion = mean of z-scores. "
             "Values: mean ± std over seeds of the category-averaged metric. `img_auroc` uses the map max, "
             "`img_auroc_top1` the mean of the top 1% pixels.", ""]
    for enc, g in df.groupby("encoder"):
        per_seed = g.groupby(["mode", "score", "seed"])[METRICS].mean()
        mu = per_seed.groupby(["mode", "score"]).mean()
        sd = per_seed.groupby(["mode", "score"]).std().fillna(0)
        lines += [f"## {enc}", "", md(mu.round(3).astype(str) + " ± " + sd.round(3).astype(str)), ""]
        # does anything beat kNN by more than its seed std?
        knn = mu.xs("knn", level="score").iloc[0]
        lines += ["### vs. kNN (mean difference; * = larger than 2× seed std and > 0.005)", ""]
        diff = mu.drop(index="knn", level="score") - knn
        sd_ = sd.drop(index="knn", level="score")
        star = ((diff.abs() > 2 * sd_) & (diff.abs() > 0.005)).replace({True: "*", False: ""})
        lines += [md(diff.round(3).astype(str) + star), ""]
        for metric in ["aupro", "px_ap", "img_auroc_top1"]:
            piv = g.groupby(["category", "mode", "score"])[metric].mean().unstack(["mode", "score"]).round(3)
            keep = [c for c in piv.columns if c[1] in ("knn", "disc", "disc+knn", "disc+resid+knn")]
            keep = sorted(keep, key=lambda c: (c[0] != "-", c))
            lines += [f"### {metric} per category (mean over seeds)", "", md(piv[keep]), ""]
    (args.out / "report_fusion.md").write_text("\n".join(lines))
    print(f"\nDone. {args.out / 'report_fusion.md'}")


if __name__ == "__main__":
    main()
