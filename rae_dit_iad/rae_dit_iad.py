#!/usr/bin/env python3
"""RAE + original DiT for industrial anomaly detection (IAD).

Trains the official RAE DiT (DiT-DH-XL) on MVTec AD *normal* images with one
of three objectives — the original flow-matching loss (Experiment A), an
image-space L1 loss through the frozen/fine-tuned RAE decoder (Experiments
B/C), or a configurable hybrid (D/E) — and evaluates reconstruction-based
anomaly localization (pixel AUROC / AP / AUPRO, image AUROC / AP).

Subcommands
-----------
``train``
    ``python rae_dit_iad.py train --config configs/B_image_l1.yaml \
        --mvtec-root $MVTEC_ROOT --out results_dit_iad --categories carpet screw``
``eval``
    ``python rae_dit_iad.py eval --config configs/B_image_l1.yaml \
        --mvtec-root $MVTEC_ROOT --out results_dit_iad --categories carpet screw \
        --mode single --t 0.5``

    With ``--config`` the checkpoints are resolved in
    ``<out>/<experiment.id>/`` (the layout ``train`` writes); otherwise pass
    ``--ckpt`` explicitly.

Everything model-side is delegated to the official RAE repo (see
``iad.models``); metrics and MVTec loading come from ``latent_draem``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# shared utilities live in the sibling latent_draem/ directory (project
# convention: the directory itself is the import root, see rae_recon_ablation.py)
_LATENT_DRAEM_DIR = HERE.parent / "latent_draem"
if str(_LATENT_DRAEM_DIR) not in sys.path:
    sys.path.insert(0, str(_LATENT_DRAEM_DIR))

from latent_draem import PILOT_CATEGORIES, evaluate_maps, load_mask, load_rgb, mvtec_split, seed_all  # noqa: E402
from iad import core, models as M  # noqa: E402
from iad import infer as I  # noqa: E402
from iad import train as T  # noqa: E402

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

IMAGE_AGGS = ("mean", "max", "top1")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_experiment_config(path: Path) -> Dict[str, Any]:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(path))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg, dict)
    return cfg


def _parse_categories(spec: Optional[str]) -> List[str]:
    if not spec or spec == "all":
        return list(ALL_CATEGORIES)
    if spec == "pilot":
        return list(PILOT_CATEGORIES)
    return [c.strip() for c in spec.split(",") if c.strip()]


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_experiment_config(Path(args.config))
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.scope is not None:
        cfg["scope"] = args.scope
    if args.lambda_fm is not None:
        cfg["experiment"]["lambda_fm"] = float(args.lambda_fm)
    if args.no_ema:
        cfg["training"]["use_ema"] = False
    seed = int(cfg.get("seed", 0))

    rae_root = M.setup_rae_root(Path(args.rae_root))
    device = _device(args.device)
    categories = _parse_categories(args.categories)
    out = Path(args.out) / cfg["experiment"]["id"]
    out.mkdir(parents=True, exist_ok=True)

    with (out / "args.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2, default=str)
    with (out / "config.resolved.json").open("w") as f:
        json.dump(cfg, f, indent=2, default=str)

    summary = T.train(
        cfg=cfg,
        rae_root=rae_root,
        mvtec_root=Path(args.mvtec_root),
        out_dir=out,
        device=device,
        categories=categories,
        seed=seed,
        scope=str(cfg.get("scope", "joint")),
        steps=args.steps,
        batch_size=args.batch_size,
        resume=Path(args.resume) if args.resume else None,
    )
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------

def _device(spec: Optional[str]) -> torch.device:
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_ckpt(out_dir: Path, spec: Optional[str], category: str, which: str) -> Path:
    """``which`` is ``best`` or ``last``.  Per-category ckpts win over joint."""
    if spec:
        p = Path(spec)
        if not p.is_absolute() and not p.exists():
            cand = out_dir / p
            if cand.exists():
                p = cand
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    for name in (f"{category}_{which}.pt", f"joint_{which}.pt"):
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no checkpoint found in {out_dir} (looked for {category}_{which}.pt / joint_{which}.pt)")


def _build_models_from_ckpt(ckpt_path: Path, rae_root: Path, device: torch.device):
    """Instantiate RAE + DiT; weights come from *our* checkpoint (not the official one)."""
    state = T.load_checkpoint(ckpt_path)
    cfg = state["config"]
    sections = M.load_official_sections(cfg["rae_config"], rae_root,
                                        cfg.get("rae_config_overrides"))
    rae_cfg, stage2_cfg, misc_cfg = M._to_dict(sections[0]), M._to_dict(sections[1]), M._to_dict(sections[5])
    misc = M._to_dict(misc_cfg)

    rae = M.build_rae(rae_root, rae_cfg, device,
                      require_decoder_ckpt=not cfg.get("allow_random_decoder", False))
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    dit = M.build_dit(stage2_cfg, rae_root, device, require_ckpt=False)
    dit.load_state_dict(state["model"], strict=True)

    use_ema = bool(cfg.get("training", {}).get("use_ema", True)) and state.get("ema") is not None
    if use_ema:
        dit.load_state_dict(state["ema"], strict=True)
    if state.get("decoder") is not None:
        rae.decoder.load_state_dict(state["decoder"], strict=True)
    dit.eval()
    return rae, dit, misc, cfg, state, use_ema


def cmd_eval(args: argparse.Namespace) -> None:
    device = _device(args.device)
    rae_root = M.setup_rae_root(Path(args.rae_root))
    categories = _parse_categories(args.categories)
    out = Path(args.out)
    if args.config:
        base_cfg = load_experiment_config(Path(args.config))
        exp_id = base_cfg["experiment"]["id"]
        # train writes checkpoints to <out>/<exp_id>/; follow that layout
        # unless the user points --ckpt somewhere explicit.
        if args.ckpt is None and (out / exp_id).is_dir():
            out = out / exp_id
    else:
        base_cfg = None
        exp_id = args.exp_id or "unknown"
    out.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None
               else (base_cfg.get("seed", 0) if base_cfg else 0))

    # inference defaults: CLI > config `inference:` section > parser hard default
    inf = dict(base_cfg.get("inference", {})) if base_cfg else {}

    def _inf_arg(cli_val, key, default):
        if cli_val is not None:
            return cli_val
        if key in inf:
            return inf[key]
        return default

    def _as_list(v):
        if isinstance(v, (list, tuple)):
            return list(v)
        return [x.strip() for x in str(v).split(",") if x.strip()]

    mode_str = _inf_arg(args.mode, "mode", "single")
    t_str = _inf_arg(args.t, "t", 0.5)
    reps = max(int(_inf_arg(args.repeats, "repeats", 1)), 1)
    ode_steps = int(_inf_arg(args.ode_steps, "ode_steps", 50))
    blur_sigma = float(_inf_arg(args.blur_sigma, "blur_sigma", 4.0))

    t_list = [float(v) for v in _as_list(t_str)]
    modes = [str(m).strip() for m in _as_list(mode_str)]

    results_csv = out / "results.csv"
    done = _read_done(results_csv)

    per_image_rows: List[Dict[str, Any]] = []
    new_rows: List[Dict[str, Any]] = []
    for cat in categories:
        ckpt_path = _resolve_ckpt(out, args.ckpt, cat, args.ckpt_which)
        rae, dit, misc, cfg, state, used_ema = _build_models_from_ckpt(ckpt_path, rae_root, device)
        exp_id = str(cfg["experiment"].get("id", exp_id))  # provenance from the ckpt itself
        num_classes = int(misc.get("num_classes", 1000))
        shift = core.time_dist_shift_from_misc(misc)
        corruption = cfg["experiment"].get("corruption", "flow")
        size = int(cfg.get("size", 256))
        seed_all(seed)

        train_paths, test_entries = mvtec_split(Path(args.mvtec_root), cat)
        del train_paths  # test split only; training data never touched here
        # test entries are (path, defect_name, mask_path) — project convention
        paths = [Path(p) for p, _defect, _m in test_entries]
        masks = (np.stack([load_mask(m, size) for _p, _d, m in test_entries])
                 if test_entries else np.zeros((0, size, size), bool))
        labels = np.array([int(d != "good") for _p, d, _m in test_entries])

        # batched inference (loading lazily per batch to bound memory)
        bs = int(args.batch_size)
        mode0, t0_setting = modes[0], t_list[0]
        maps_all: List[np.ndarray] = []
        recons_uint8: List[np.ndarray] = []
        inputs_uint8: List[np.ndarray] = []
        infer_time = 0.0
        dit_passes = 0
        mode_used = None
        t_used = None
        for start in range(0, len(paths), bs):
            chunk = paths[start:start + bs]
            x = torch.stack([load_rgb(p, size) for p in chunk]).to(device)
            idxs = list(range(start, start + len(chunk)))
            t0 = time.time()
            xhat, maps, meta = I.reconstruct(
                rae, dit, x, idxs,
                mode=mode0, t=t0_setting,
                repeats=reps, ode_steps=ode_steps,
                num_classes=num_classes, corruption=corruption, seed=seed,
                time_dist_shift=shift,
            )
            infer_time += time.time() - t0
            dit_passes = meta["dit_passes"]
            mode_used, t_used = meta["mode"], meta["t"]
            maps_all.append(maps.cpu().numpy())
            inputs_uint8.append((x.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))
            recons_uint8.append((xhat.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))

        if not paths:
            continue
        maps_np = np.concatenate(maps_all, axis=0)
        inputs_np = np.concatenate(inputs_uint8, axis=0)
        recons_np = np.concatenate(recons_uint8, axis=0)

        key_base = dict(exp=exp_id, cat=cat, seed=seed, reps=reps,
                        ckpt=ckpt_path.name, ema=int(used_ema))
        row = _score_setting(maps_np, masks, labels, size, blur_sigma)
        row.update(key_base, mode=mode_used, t=t_used,
                   n_images=len(paths),
                   infer_s_per_img=round(infer_time / max(len(paths), 1), 4),
                   dit_passes_per_img=dit_passes)
        sig = _row_sig(row)
        if sig not in done:
            new_rows.append(row)
            done.add(sig)
            per_image_rows.extend(_per_image_rows(
                paths, labels, maps_np, exp_id=exp_id, cat=cat, seed=seed,
                mode=mode_used, t=t_used))
        if int(args.viz) > 0:
            viz_path = out / "viz" / f"{cat}_{exp_id}_{mode_used}_t{t_used}.png"
            n_normal = max(1, int(args.viz) // 3)
            I.save_viz(inputs_np, recons_np, maps_np, masks, labels, viz_path,
                       n_normal=n_normal)
        print(f"[eval/{cat}] {ckpt_path.name} mode={mode_used} t={t_used} reps={reps} "
              f"I-AUROC(max)={row['img_auroc_max']:.4f} P-AUROC={row['px_auroc']:.4f} "
              f"AUPRO={row['aupro']:.4f} ({infer_time / max(len(paths), 1):.3f}s/img)", flush=True)

        # ---- additional (mode, t) settings for sweeps --------------------------
        for extra_mode in modes:
            for extra_t in t_list:
                if extra_mode == mode_used and extra_t == t_used:
                    continue
                # skip settings already in results.csv (no wasted inference)
                cand_sig = (exp_id, cat, seed, ckpt_path.name, int(used_ema),
                            extra_mode, extra_t, reps, blur_sigma)
                if cand_sig in done:
                    continue
                row_x = _eval_setting(rae, dit, paths, masks, labels, size, blur_sigma,
                                      mode=extra_mode, t=extra_t, reps=reps,
                                      ode_steps=ode_steps, num_classes=num_classes,
                                      corruption=corruption, seed=seed, shift=shift,
                                      device=device, bs=bs)
                if row_x is None:
                    continue
                row_x, maps_x = row_x
                row_x.update(key_base, mode=extra_mode, t=extra_t, n_images=len(paths))
                sig = _row_sig(row_x)
                if sig not in done:
                    new_rows.append(row_x)
                    done.add(sig)
                    per_image_rows.extend(_per_image_rows(
                        paths, labels, maps_x, exp_id=exp_id, cat=cat, seed=seed,
                        mode=extra_mode, t=extra_t))
                print(f"[eval/{cat}] extra mode={extra_mode} t={extra_t} "
                      f"I-AUROC(max)={row_x['img_auroc_max']:.4f} "
                      f"P-AUROC={row_x['px_auroc']:.4f} AUPRO={row_x['aupro']:.4f}", flush=True)
        del rae, dit

    if new_rows:
        _append_rows(results_csv, new_rows)
    pi_path = out / "per_image.csv"
    _append_rows(pi_path, per_image_rows) if pi_path.exists() else _write_rows(pi_path, per_image_rows)
    write_report(results_csv, out / "report.md", exp_id)
    print(f"wrote {results_csv} ({len(new_rows)} new rows), report -> {out / 'report.md'}")


def _per_image_rows(paths, labels, maps_np, *, exp_id, cat, seed, mode, t) -> List[Dict[str, Any]]:
    return [{
        "exp": exp_id, "cat": cat, "seed": seed, "mode": mode, "t": t,
        "image": p.name, "label": int(labels[i]),
        **{f"score_{a}": float(I.image_score(maps_np[i:i + 1], a)[0])
           for a in IMAGE_AGGS},
    } for i, p in enumerate(paths)]


def _eval_setting(rae, dit, paths, masks, labels, size, blur_sigma, *, mode, t, reps,
                  ode_steps, num_classes, corruption, seed, shift, device, bs):
    """Run one (mode, t) setting. Returns ``(row, maps_np)`` or ``None``."""
    maps_list = []
    infer_time = 0.0
    dit_passes = 0
    for start in range(0, len(paths), bs):
        chunk = paths[start:start + bs]
        x = torch.stack([load_rgb(p, size) for p in chunk]).to(device)
        idxs = list(range(start, start + len(chunk)))
        t0 = time.time()
        _, maps, meta = I.reconstruct(
            rae, dit, x, idxs, mode=mode, t=t, repeats=reps, ode_steps=ode_steps,
            num_classes=num_classes, corruption=corruption, seed=seed,
            time_dist_shift=shift)
        infer_time += time.time() - t0
        dit_passes = meta["dit_passes"]
        maps_list.append(maps.cpu().numpy())
    if not maps_list:
        return None
    maps_np = np.concatenate(maps_list, axis=0)
    row = _score_setting(maps_np, masks, labels, size, blur_sigma)
    row.update(infer_s_per_img=round(infer_time / max(len(paths), 1), 4),
               dit_passes_per_img=dit_passes)
    return row, maps_np


def _score_setting(maps_np: np.ndarray, masks: np.ndarray, labels: np.ndarray,
                   size: int, blur_sigma: float) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    blurred = I.blur_maps(maps_np, float(blur_sigma))
    px = evaluate_maps(blurred, masks, labels, do_pro=True)
    row: Dict[str, Any] = {
        "px_auroc": px["px_auroc"], "px_ap": px["px_ap"], "aupro": px["aupro"],
        "blur_sigma": float(blur_sigma),
    }
    two = labels.min() != labels.max()
    for agg in IMAGE_AGGS:
        scores = I.image_score(blurred, agg)
        row[f"img_auroc_{agg}"] = float(roc_auc_score(labels, scores)) if two else float("nan")
        row[f"img_ap_{agg}"] = float(average_precision_score(labels, scores)) if two else float("nan")
    return row


# ---------------------------------------------------------------------------
# results table / report
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "exp", "cat", "seed", "ckpt", "ema", "mode", "t", "reps",
    "blur_sigma", "n_images", "img_auroc_mean", "img_auroc_max", "img_auroc_top1",
    "img_ap_mean", "img_ap_max", "img_ap_top1", "px_auroc", "px_ap", "aupro",
    "infer_s_per_img", "dit_passes_per_img",
]


def _row_sig(row: Dict[str, Any]) -> tuple:
    return (row.get("exp"), row.get("cat"), row.get("seed"), row.get("ckpt"),
            row.get("ema"), row.get("mode"), row.get("t"), row.get("reps"),
            row.get("blur_sigma"))


def _read_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            done.add((r.get("exp"), r.get("cat"), int(r.get("seed", 0)), r.get("ckpt"),
                      int(r.get("ema", 0)), r.get("mode"), float(r.get("t", 0)),
                      int(r.get("reps", 1)), float(r.get("blur_sigma", 0))))
    return done


def _write_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = _union_fields(rows)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = _union_fields(rows)
    if csv_path.exists():
        with csv_path.open() as f:
            existing = next(csv.reader(f), None)
        if existing:
            fields = list(dict.fromkeys(existing + fields))
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if csv_path.stat().st_size == 0:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _union_fields(rows: List[Dict[str, Any]]) -> List[str]:
    seen: Dict[str, None] = {}
    for r in rows:
        for k in r:
            seen.setdefault(k)
    order = [k for k in RESULT_FIELDS if k in seen]
    order += [k for k in seen if k not in order]
    return order


def write_report(csv_path: Path, report_path: Path, exp_id: str) -> None:
    """Aggregate + per-category markdown tables from results.csv."""
    if not csv_path.exists():
        return
    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("exp") == exp_id or not exp_id]
    if not rows:
        return

    metrics = ["img_auroc_mean", "img_auroc_max", "img_auroc_top1",
               "img_ap_mean", "img_ap_max", "img_ap_top1",
               "px_auroc", "px_ap", "aupro"]
    groups: Dict[tuple, List[Dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r.get("mode"), r.get("t"), r.get("reps"), r.get("ckpt")), []).append(r)

    def mean(vals: List[float]) -> float:
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def fmt(r: Dict[str, Any], m: str) -> str:
        v = r.get(m)
        if v in (None, ""):
            return "n/a"
        try:
            return f"{float(v):.4f}"
        except ValueError:
            return str(v)

    config_cols = ["mode", "t", "reps", "ckpt"]

    def table_header(cols: List[str]) -> List[str]:
        hdr = "| " + " | ".join(cols + metrics) + " |"
        return [hdr, "|" + "---|" * (len(cols) + len(metrics))]

    lines = [f"# RAE-DiT IAD results ({exp_id})", "",
             "Aggregate = mean over categories. Image AUROC/AP reported for all three",
             "aggregations (mean / max / top-1% pixel scores).", "", "## Aggregate", ""]
    lines += table_header(config_cols)
    for key, rs in sorted(groups.items()):
        mode, t, reps, ckpt = key
        cells = []
        for m in metrics:
            vals = [float(r[m]) for r in rs if r.get(m) not in (None, "")]
            cells.append(f"{mean(vals):.4f}" if vals else "n/a")
        lines.append(f"| {mode} | {t} | {reps} | {ckpt} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-category", ""]
    lines += table_header(["cat"] + config_cols)
    for r in sorted(rows, key=lambda r: (r.get("ckpt", ""), r.get("mode", ""),
                                         float(r.get("t", 0)), r.get("cat", ""))):
        head = [str(r.get("cat")), str(r.get("mode")), str(r.get("t")),
                str(r.get("reps")), str(r.get("ckpt"))]
        lines.append("| " + " | ".join(head + [fmt(r, m) for m in metrics]) + " |")
    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train the DiT on MVTec normal images")
    tr.add_argument("--config", required=True, help="experiment YAML (configs/*.yaml)")
    tr.add_argument("--rae-root", default=str(HERE.parent / "rae_fidelity" / "RAE"))
    tr.add_argument("--mvtec-root",
                    default="/home/maometus/Documents/datasets/mvtec_anomaly_detection")
    tr.add_argument("--out", default=str(HERE / "results_dit_iad"))
    tr.add_argument("--categories", default="pilot",
                    help="'all' | 'pilot' | comma list")
    tr.add_argument("--scope", choices=["joint", "per_category"], default=None)
    tr.add_argument("--seed", type=int, default=None)
    tr.add_argument("--steps", type=int, default=None)
    tr.add_argument("--batch-size", type=int, default=None)
    tr.add_argument("--lambda-fm", type=float, default=None,
                    help="override experiment.lambda_fm (hybrid ablations)")
    tr.add_argument("--resume", default=None, help="checkpoint to resume from")
    tr.add_argument("--no-ema", action="store_true")
    tr.add_argument("--device", default=None)

    ev = sub.add_parser("eval", help="evaluate anomaly detection metrics")
    ev.add_argument("--config", default=None,
                    help="optional; ckpt's embedded config is used when omitted")
    ev.add_argument("--exp-id", default=None, help="label when --config is omitted")
    ev.add_argument("--ckpt", default=None,
                    help="checkpoint path (default: auto-resolve <cat>_best/joint_best in --out)")
    ev.add_argument("--ckpt-which", choices=["best", "last"], default="best")
    ev.add_argument("--rae-root", default=str(HERE.parent / "rae_fidelity" / "RAE"))
    ev.add_argument("--mvtec-root",
                    default="/home/maometus/Documents/datasets/mvtec_anomaly_detection")
    ev.add_argument("--out", default=str(HERE / "results_dit_iad"))
    ev.add_argument("--categories", default="all")
    ev.add_argument("--seed", type=int, default=None)
    ev.add_argument("--mode", default=None,
                    help="single | multi | ode (comma list for sweeps)")
    ev.add_argument("--t", default=None, help="noise level(s), comma list")
    ev.add_argument("--repeats", type=int, default=None, help="repetitions for mode=multi")
    ev.add_argument("--ode-steps", type=int, default=None)
    ev.add_argument("--blur-sigma", type=float, default=None)
    ev.add_argument("--batch-size", type=int, default=16)
    ev.add_argument("--viz", type=int, default=6, help="grid images per category (0=off)")
    ev.add_argument("--device", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
