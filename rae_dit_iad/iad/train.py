"""Training engine for the RAE + DiT IAD experiments (losses A/B/C/D/E).

Reuses the official RAE training conventions where they exist:

* AdamW with official hyperparameters and the official linear
  warmup/decay ``LambdaLR`` (``utils.optim_utils``).
* EMA via official ``update_ema`` (decay 0.9995 by default).
* ``clip_grad`` = 1.0, ``precision`` fp32/fp16/bf16 autocast like ``train.py``.
* Time/noise sampling and the flow path live in ``iad.core`` (which calls the
  official ``Transport`` primitives).

Deviations from the official ``train.py`` (documented in README):

* single-process (no DDP), step-based budget instead of epochs;
* class labels are the official CFG **null token** (MVTec has no ImageNet
  label), a no-op w.r.t. the official label dropout;
* the encoder is frozen and its latents are pre-cached (encoder is never
  updated, so caching is exact);
* batch sampling + validation use saved RNG states for exact resume.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from latent_draem import load_rgb, mvtec_split, seed_all

from iad import core
from iad import models as M


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def split_train_val(
    mvtec_root: Path,
    categories: Sequence[str],
    size: int,
    n_val: int,
    val_seed: int = 1234,
) -> Tuple[List[Path], List[Path], Dict[str, List[Path]]]:
    """Held-out *normal* validation images (no test images, no leakage).

    Mirrors ``rae_recon_ablation.py``: per category, a ``np.random.default_rng(1234)``
    permutation of ``train/good``; the first ``n_val`` paths are validation,
    the rest training.  Test images are never used here.
    """
    train_paths: List[Path] = []
    val_paths: List[Path] = []
    per_cat: Dict[str, List[Path]] = {}
    for cat in categories:
        tr, _te = mvtec_split(Path(mvtec_root), cat)
        tr = [Path(p) for p in tr]
        if n_val > 0 and len(tr) > n_val:
            perm = np.random.default_rng(val_seed).permutation(len(tr))
            val = [tr[i] for i in perm[:n_val]]
            keep = [tr[i] for i in perm[n_val:]]
        else:
            val, keep = [], tr
        per_cat[cat] = keep
        train_paths.extend(keep)
        val_paths.extend(val)
    return train_paths, val_paths, per_cat


def preload_images(paths: Sequence[Path], size: int) -> Dict[Path, torch.Tensor]:
    """Load images once (fp16 CPU, ~1.2 GB for the full MVTec train split)."""
    out: Dict[Path, torch.Tensor] = {}
    for p in paths:
        out[Path(p)] = load_rgb(Path(p), size).half()
    return out


@torch.no_grad()
def encode_cache(
    rae: nn.Module,
    paths: Sequence[Path],
    size: int,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[Path, torch.Tensor]:
    """Pre-encode latents with the frozen encoder (exact: encoder never trains)."""
    rae.eval()
    out: Dict[Path, torch.Tensor] = {}
    for i in range(0, len(paths), batch_size):
        chunk = [Path(p) for p in paths[i:i + batch_size]]
        x = torch.stack([load_rgb(p, size) for p in chunk]).to(device)
        z = rae.encode(x)
        for p, zi in zip(chunk, z):
            out[p] = zi.detach().float().cpu()
    return out


def _gather(
    paths: Sequence[Path],
    images: Dict[Path, torch.Tensor],
    latents: Optional[Dict[Path, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    x = torch.stack([images[Path(p)] for p in paths]).to(device=device, dtype=torch.float32)
    z = None
    if latents:
        z = torch.stack([latents[Path(p)] for p in paths]).to(device)
    return x, z


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def _rng_states() -> Dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _load_rng_states(states: Dict[str, Any]) -> None:
    torch.set_rng_state(states["torch"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(states["cuda"])
        except RuntimeError:
            pass  # device count changed between save and resume
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])


def save_checkpoint(
    path: Path,
    step: int,
    dit: nn.Module,
    ema: Optional[nn.Module],
    rae: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    state = {
        "step": step,
        "model": dit.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "decoder": rae.decoder.state_dict()
        if cfg.get("experiment", {}).get("decoder_finetune") else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": cfg,
        "meta": meta,
        "rng": _rng_states(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    # our own checkpoints (numpy/python RNG state, config dicts) are not
    # weights-only; torch>=2.6 defaults to weights_only=True, so opt out.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


def apply_checkpoint(
    state: Dict[str, Any],
    dit: nn.Module,
    ema: Optional[nn.Module],
    rae: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
) -> int:
    dit.load_state_dict(state["model"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    if state.get("decoder") is not None:
        rae.decoder.load_state_dict(state["decoder"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and state.get("rng") is not None:
        _load_rng_states(state["rng"])
    return int(state.get("step", 0))


class _RngGuard:
    """Run validation under a fixed seed without perturbing the training stream."""

    def __enter__(self):
        self._states = _rng_states()
        return self

    def __exit__(self, *exc):
        _load_rng_states(self._states)
        return False


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False), None
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested but the CUDA device does not support it")
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
    return torch.autocast(device_type="cuda", dtype=dtype), scaler


@torch.no_grad()
def validate(
    rae: nn.Module,
    dit: nn.Module,
    transport,
    val_paths: Sequence[Path],
    images: Dict[Path, torch.Tensor],
    latents: Dict[Path, torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    val_seed: int,
    num_classes: int,
) -> Dict[str, float]:
    """Deterministic validation: identical noise draws on every call.

    Draws are fixed by re-seeding to ``val_seed`` inside an RNG guard, so the
    metric is comparable across steps (and unaffected on the training stream).
    """
    if not val_paths:
        return {}
    exp = cfg["experiment"]
    noise_cfg = cfg["noise"]
    was_training = dit.training
    dit.eval()
    totals: Dict[str, List[float]] = {}
    with _RngGuard():
        seed_all(val_seed)
        for i in range(0, len(val_paths), batch_size):
            chunk = list(val_paths[i:i + batch_size])
            x, z = _gather(chunk, images, latents, device)
            if z is None:  # cache_latents=false
                with torch.no_grad():
                    z = rae.encode(x)
            y = M.null_labels(len(chunk), num_classes, device)
            loss, terms, _, _ = core.compute_losses(
                rae, dit, transport, x, z,
                loss_type=exp["loss"], noise_cfg=noise_cfg,
                lambda_fm=float(exp.get("lambda_fm", 0.0)), y=y,
                corruption=exp.get("corruption", "flow"),
                size=int(cfg.get("size", 256)),
                clamp_recon=bool(exp.get("clamp_recon", False)),
            )
            for k, v in terms.items():
                totals.setdefault(k, []).append(v)
            totals.setdefault("total", []).append(float(loss.detach()))
    if was_training:
        dit.train()
    return {f"val_{k}": float(np.mean(v)) for k, v in totals.items()}


def train(
    cfg: Dict[str, Any],
    rae_root: Path,
    mvtec_root: Path,
    out_dir: Path,
    device: torch.device,
    categories: Sequence[str],
    seed: int = 0,
    scope: str = "joint",
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    resume: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one experiment (or 15 per-category runs when ``scope='per_category'``).

    Returns a summary dict; writes checkpoints + ``metrics.jsonl`` under
    ``out_dir``.
    """
    exp = cfg["experiment"]
    tcfg = dict(cfg["training"])
    if steps is not None:
        tcfg["steps"] = int(steps)
    if batch_size is not None:
        tcfg["batch_size"] = int(batch_size)
    n_steps = int(tcfg.get("steps", 2000))
    bs = int(tcfg.get("batch_size", 8))
    precision = str(tcfg.get("precision", "fp32"))
    clip_grad = tcfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad) if clip_grad and float(clip_grad) > 0 else None
    log_every = int(tcfg.get("log_every", 50))
    val_every = int(tcfg.get("val_every", 250))
    save_every = int(tcfg.get("save_every", 500))
    n_val = int(tcfg.get("n_val", 20))
    size = int(cfg.get("size", 256))
    use_ema = bool(tcfg.get("use_ema", True))
    ema_decay = float(tcfg.get("ema_decay", 0.9995))
    cache = bool(tcfg.get("cache_latents", True))
    loss_type = exp["loss"]
    corruption = exp.get("corruption", "flow")
    lambda_fm = float(exp.get("lambda_fm", 0.0))
    if loss_type == "hybrid" and corruption != "flow":
        raise ValueError("hybrid loss requires corruption='flow'")
    best_metric = exp.get("best_metric") or ("fm" if loss_type == "flow_matching" else "img")

    seed_all(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- official components -------------------------------------------------
    sections = M.load_official_sections(cfg["rae_config"], rae_root,
                                        cfg.get("rae_config_overrides"))
    rae_cfg, stage2_cfg, transport_cfg, misc_cfg = sections[0], sections[1], sections[2], sections[5]
    misc = M._to_dict(misc_cfg)
    num_classes = int(misc.get("num_classes", 1000))

    from utils.optim_utils import build_scheduler  # official

    # Initial DiT state (official released checkpoint) for per-category resets.
    rae = M.build_rae(rae_root, M._to_dict(rae_cfg), device,
                      require_decoder_ckpt=not cfg.get("allow_random_decoder", False))
    M.freeze_encoder(rae)
    init_decoder_state = copy.deepcopy(rae.decoder.state_dict())
    dit = M.build_dit(M._to_dict(stage2_cfg), rae_root, device,
                      require_ckpt=bool(cfg.get("init", {}).get("from_pretrained", True)))
    init_state = copy.deepcopy(dit.state_dict())
    transport = M.build_transport(M._to_dict(transport_cfg), misc)

    scopes = [(c, [c]) for c in categories] if scope == "per_category" else [("joint", list(categories))]
    summaries = []
    for scope_name, cats in scopes:
        if scope == "per_category":
            # fresh DiT initialization + pristine decoder for every category
            dit.load_state_dict(init_state)
            rae.decoder.load_state_dict(init_decoder_state)
            if exp.get("decoder_finetune"):
                M.unfreeze_decoder(rae)
            else:
                M.freeze_decoder(rae)
        elif not exp.get("decoder_finetune"):
            M.freeze_decoder(rae)

        s = _train_scope(
            cfg=cfg, tcfg=tcfg, cats=cats, scope_name=scope_name, seed=seed,
            rae=rae, dit=dit, transport=transport,
            rae_root=rae_root, mvtec_root=Path(mvtec_root), out_dir=out_dir,
            device=device, misc=misc, num_classes=num_classes,
            n_steps=n_steps, bs=bs, precision=precision, clip_grad=clip_grad,
            log_every=log_every, val_every=val_every, save_every=save_every,
            n_val=n_val, size=size, use_ema=use_ema, ema_decay=ema_decay,
            cache=cache, loss_type=loss_type, corruption=corruption,
            lambda_fm=lambda_fm, best_metric=best_metric, resume=resume,
            build_scheduler=build_scheduler,
        )
        summaries.append(s)
    return {"runs": summaries}


def _train_scope(
    *, cfg, tcfg, cats, scope_name, seed, rae, dit, transport,
    rae_root, mvtec_root, out_dir, device, misc, num_classes,
    n_steps, bs, precision, clip_grad, log_every, val_every, save_every,
    n_val, size, use_ema, ema_decay, cache, loss_type, corruption, lambda_fm,
    best_metric, resume, build_scheduler,
) -> Dict[str, Any]:
    exp = cfg["experiment"]
    tag = scope_name
    accum = max(int(tcfg.get("grad_accum_steps", 1)), 1)
    effective_bs = bs * accum

    train_paths, val_paths, _ = split_train_val(mvtec_root, cats, size, n_val)
    if not train_paths:
        raise RuntimeError(f"No training images for categories {cats} under {mvtec_root}")
    print(f"[iad/{tag}] {len(train_paths)} train / {len(val_paths)} val normal images "
          f"(cats: {','.join(cats)})", flush=True)
    print(f"[iad/{tag}] batch {bs} x {accum} accumulation = effective {effective_bs}",
          flush=True)

    images = preload_images(sorted(set(train_paths) | set(val_paths)), size)
    latents = encode_cache(rae, sorted(set(train_paths) | set(val_paths)), size, device) \
        if cache else None

    # ---- optimizer / scheduler / EMA ----------------------------------------
    optimizer, base_lr, dec_lr, opt_msg = M.build_optimizer(
        dit, rae, bool(exp.get("decoder_finetune")), tcfg)
    scheduler = None
    if tcfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, 1, tcfg)
        M.fix_param_group_lrs(optimizer, base_lr, dec_lr, scheduler)
        print(f"[iad/{tag}] {sched_msg}", flush=True)
    else:
        sched_msg = "No LR scheduler."
    print(f"[iad/{tag}] {opt_msg}", flush=True)

    ema = None
    if use_ema:
        ema = copy.deepcopy(dit)
        ema.requires_grad_(False)
        ema.eval()

    start_step = 0
    if resume is not None:
        state = load_checkpoint(Path(resume))
        start_step = apply_checkpoint(state, dit, ema, rae, optimizer, scheduler)
        print(f"[iad/{tag}] resumed from {resume} at step {start_step}", flush=True)

    autocast_ctx, scaler = _autocast(device, precision)
    trainable = list(dit.parameters())
    if exp.get("decoder_finetune"):
        trainable = trainable + list(rae.decoder.parameters())

    meta = {
        "categories": list(cats),
        "scope": cfg.get("scope", "joint"),
        "seed": seed,
        "rae_config": str(cfg["rae_config"]),
        "loss": loss_type,
        "corruption": corruption,
        "lambda_fm": lambda_fm,
        "batch_micro": bs,
        "grad_accum_steps": accum,
        "batch_effective": effective_bs,
        "size": size,
        "init": cfg.get("init", {}),
        "null_label": num_classes,
        "encoder": "stage1.RAE (frozen; weights referenced by rae_config)",
    }

    def ckpt_paths():
        prefix = tag if tag != "joint" else "joint"
        return out_dir / f"{prefix}_last.pt", out_dir / f"{prefix}_best.pt"

    last_path, best_path = ckpt_paths()
    metrics_f = out_dir / "metrics.jsonl"
    best_val = math.inf
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)  # batch indices
    running: Dict[str, List[float]] = {}
    t_last = time.time()
    ema_model = ema if ema is not None else dit

    step = start_step
    while step < n_steps:
        dit.train()
        if exp.get("decoder_finetune"):
            rae.decoder.train()

        optimizer.zero_grad(set_to_none=True)
        micro_terms: Dict[str, List[float]] = {}
        for _micro in range(accum):
            idx = torch.randint(0, len(train_paths), (bs,), generator=rng).tolist()
            chunk = [train_paths[i] for i in idx]
            x, z = _gather(chunk, images, latents, device)
            if z is None:
                with torch.no_grad():
                    z = rae.encode(x)
            y = M.null_labels(len(chunk), num_classes, device)

            with autocast_ctx:
                total, terms, _, _ = core.compute_losses(
                    rae, dit, transport, x, z,
                    loss_type=loss_type, noise_cfg=cfg["noise"],
                    lambda_fm=lambda_fm, y=y, corruption=corruption,
                    size=size, clamp_recon=bool(exp.get("clamp_recon", False)),
                )
            # gradient accumulation: divide so the accumulated gradient is the
            # mean over the effective batch
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total / accum).backward()
            else:
                (total / accum).backward()
            for k, v in terms.items():
                micro_terms.setdefault(k, []).append(v)

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)  # grads must be unscaled before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, clip_grad if clip_grad else math.inf)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, clip_grad if clip_grad else math.inf)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            from utils.train_utils import update_ema  # official
            with torch.no_grad():
                update_ema(ema, dit, decay=ema_decay)

        for k, vals in micro_terms.items():
            running.setdefault(k, []).append(float(np.mean(vals)))
        step += 1

        if step % log_every == 0 or step >= n_steps:
            dt = max(time.time() - t_last, 1e-9)
            n = len(next(iter(running.values()))) if running else log_every
            rec: Dict[str, Any] = {
                "step": step,
                "scope": tag,
                **{f"loss_{k}": float(np.mean(v)) for k, v in running.items()},
                "lr_dit": optimizer.param_groups[0]["lr"],
                "lr_dec": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None,
                "grad_norm": float(grad_norm),
                "batch_eff": effective_bs,
                "step_time_s": dt / max(n, 1),
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                                if device.type == "cuda" else None),
                "wall_s": dt,
            }
            with metrics_f.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            msg = " ".join(f"{k}={v:.4f}" for k, v in rec.items()
                           if isinstance(v, float) and (k.startswith("loss") or k.startswith("lr")))
            print(f"[iad/{tag}] step {step}/{n_steps} {msg} "
                  f"({rec['step_time_s']:.2f}s/step)", flush=True)
            running = {}
            t_last = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if val_every > 0 and (step % val_every == 0 or step >= n_steps) and val_paths:
            vals = validate(rae, ema_model, transport, val_paths, images,
                            latents, cfg, device, bs, seed + 42, num_classes)
            rec = {"step": step, "scope": tag, **vals}
            with metrics_f.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[iad/{tag}] step {step} " +
                  " ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
            key = f"val_{best_metric}"
            if key in vals and vals[key] < best_val:
                best_val = vals[key]
                save_checkpoint(best_path, step, dit, ema, rae, optimizer,
                                scheduler, cfg, meta)
                print(f"[iad/{tag}] new best {key}={best_val:.4f} -> {best_path}", flush=True)

        if save_every > 0 and step % save_every == 0:
            save_checkpoint(last_path, step, dit, ema, rae, optimizer,
                            scheduler, cfg, meta)

    save_checkpoint(last_path, step, dit, ema, rae, optimizer, scheduler, cfg, meta)
    if best_val == math.inf and val_paths:
        save_checkpoint(best_path, step, dit, ema, rae, optimizer,
                        scheduler, cfg, meta)

    # explicit post-conditions (spec §6 / §2.3)
    enc_frozen = M.encoder_status(rae)
    print(f"[iad/{tag}] done at step {step}; encoder_frozen={enc_frozen}; "
          f"decoder_finetune={bool(exp.get('decoder_finetune'))}; best_{best_metric}="
          f"{best_val if best_val is not math.inf else 'n/a'}", flush=True)

    del images, latents
    return {
        "scope": tag,
        "steps": step,
        "best_val": None if best_val == math.inf else best_val,
        "encoder_frozen": enc_frozen,
        "last_ckpt": str(last_path),
        "best_ckpt": str(best_path) if best_path.exists() else None,
    }
