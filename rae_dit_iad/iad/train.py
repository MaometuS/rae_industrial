"""Training engine for the RAE + DiT IAD experiments (losses A/B/C/D/E/F/G).

Reuses the official RAE training conventions where they exist:

* AdamW with official hyperparameters and the official linear
  warmup/decay ``LambdaLR`` (``utils.optim_utils``).
* EMA via official ``update_ema`` (decay 0.9995 by default; the optional
  feature adaptor is EMA'd with the same decay).
* ``clip_grad`` = 1.0, ``precision`` fp32/fp16/bf16 autocast like ``train.py``.
* Time/noise sampling and the flow path live in ``iad.core`` (which calls the
  official ``Transport`` primitives).

Deviations from the official ``train.py`` (documented in README):

* single-process (no DDP), step-based budget instead of epochs;
* class labels are the official CFG **null token** (MVTec has no ImageNet
  label), a no-op w.r.t. the official label dropout;
* the encoder is frozen and its latents are pre-cached (encoder is never
  updated, so caching is exact);
* batch sampling + validation use saved RNG states for exact resume;
* experiments F/G add a small trainable feature adaptor between the frozen
  encoder and the DiT/decoder (``iad.adaptor``).
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

from iad import adaptor as A
from iad import core
from iad import disc as DS
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
    rae.encoder.to(device)  # may have been offloaded after a previous scope's cache
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
    adaptor: Optional[nn.Module] = None,
    ema_adaptor: Optional[nn.Module] = None,
    disc: Optional[nn.Module] = None,
    ema_disc: Optional[nn.Module] = None,
    eval_only: bool = False,
) -> None:
    """``eval_only``: what evaluation reads and nothing else -- the EMA weights
    (raw DiT weights only if there is no EMA), no optimizer / scheduler state.
    ~3.4 GB instead of ~13 GB for the 840M DiT; such a file cannot be resumed."""
    state = {
        "step": step,
        "model": None if (eval_only and ema is not None) else dit.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "decoder": rae.decoder.state_dict()
        if cfg.get("experiment", {}).get("decoder_finetune") else None,
        "adaptor": adaptor.state_dict() if adaptor is not None else None,
        "ema_adaptor": ema_adaptor.state_dict() if ema_adaptor is not None else None,
        "disc": disc.state_dict() if disc is not None else None,
        "ema_disc": ema_disc.state_dict() if ema_disc is not None else None,
        "optimizer": None if eval_only else optimizer.state_dict(),
        "scheduler": None if (eval_only or scheduler is None) else scheduler.state_dict(),
        "config": cfg,
        "meta": meta,
        "rng": _rng_states(),
        "slim": bool(eval_only),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def slim_checkpoint(path: Path) -> Tuple[float, float]:
    """Drop the optimizer and scheduler state from a finished checkpoint.

    Evaluation never reads them (and they are ~2/3 of an 840M-DiT file); a
    slimmed checkpoint can no longer be resumed.  Returns sizes in GB
    (before, after)."""
    import os
    path = Path(path)
    before = path.stat().st_size / 1e9
    state = load_checkpoint(path, mmap=True)
    state.pop("optimizer", None)
    state.pop("scheduler", None)
    state["slim"] = True
    tmp = path.with_suffix(".slim.tmp")
    torch.save(state, tmp)
    del state
    os.replace(tmp, path)
    return before, path.stat().st_size / 1e9


def load_checkpoint(path: Path, map_location: str = "cpu", mmap: bool = False) -> Dict[str, Any]:
    # our own checkpoints (numpy/python RNG state, config dicts) are not
    # weights-only; torch>=2.6 defaults to weights_only=True, so opt out.
    # mmap=True (eval): tensors stay on disk until touched, so the optimizer
    # state (~2/3 of the file) never enters RAM.
    if mmap:
        try:
            return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
        except (TypeError, RuntimeError):  # old torch / legacy file format: load normally
            pass
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
    adaptor: Optional[nn.Module] = None,
    ema_adaptor: Optional[nn.Module] = None,
    disc: Optional[nn.Module] = None,
    ema_disc: Optional[nn.Module] = None,
) -> int:
    dit.load_state_dict(state["model"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    if state.get("decoder") is not None:
        rae.decoder.load_state_dict(state["decoder"])
    if adaptor is not None and state.get("adaptor") is not None:
        adaptor.load_state_dict(state["adaptor"])
    if ema_adaptor is not None and state.get("ema_adaptor") is not None:
        ema_adaptor.load_state_dict(state["ema_adaptor"])
    if disc is not None and state.get("disc") is not None:
        disc.load_state_dict(state["disc"])
    if ema_disc is not None and state.get("ema_disc") is not None:
        ema_disc.load_state_dict(state["ema_disc"])
    if optimizer is not None and state.get("slim"):
        raise ValueError("this checkpoint was slimmed after training (no optimizer "
                         "state) and cannot be resumed; it is for evaluation only")
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
def modules_to_move(modules: Sequence[nn.Module], device: torch.device) -> List[nn.Module]:
    """Modules validation must bring onto ``device`` and send back to the CPU.

    Only CPU-resident modules (``ema_device: cpu``) qualify.  Comparing devices
    directly is wrong: parameters on ``cuda:0`` != ``torch.device('cuda')``, which
    used to send the GPU EMA to the CPU after the first validation and made every
    later EMA update ~8x slower (job 152776: 1.2 s -> 10 s per step from step 250).
    """
    if device.type == "cpu":
        return []
    return [m for m in modules if next(iter(m.parameters())).device.type == "cpu"]


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
    adaptor: Optional[nn.Module] = None,
    disc: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Deterministic validation: identical noise draws on every call.

    Draws are fixed by re-seeding to ``val_seed`` inside an RNG guard, so the
    metric is comparable across steps (and unaffected on the training stream).

    ``dit`` / ``adaptor`` here are whatever weights should be scored (the EMA
    copies when EMA is on); CPU-resident modules are moved to ``device`` for
    the pass and back afterwards.
    """
    if not val_paths:
        return {}
    exp = cfg["experiment"]
    noise_cfg = cfg["noise"]
    was_training = dit.training
    was_training_ad = adaptor.training if adaptor is not None else False
    dit.eval()
    if adaptor is not None:
        adaptor.eval()
    was_training_ds = disc.training if disc is not None else False
    if disc is not None:
        disc.eval()
    modules = [dit] + [m for m in (adaptor, disc) if m is not None]
    moved = modules_to_move(modules, device)
    for m in moved:
        m.to(device)
    totals: Dict[str, List[float]] = {}
    try:
        with _RngGuard():
            seed_all(val_seed)
            for i in range(0, len(val_paths), batch_size):
                chunk = list(val_paths[i:i + batch_size])
                x, z = _gather(chunk, images, latents, device)
                if z is None:  # cache_latents=false
                    with torch.no_grad():
                        z = rae.encode(x)
                y = M.null_labels(len(chunk), num_classes, device)
                _, terms, _, _ = core.compute_losses(
                    rae, dit, transport, x, z,
                    loss_type=exp["loss"], noise_cfg=noise_cfg,
                    lambda_fm=float(exp.get("lambda_fm", 0.0)), y=y,
                    corruption=exp.get("corruption", "simplenet"),
                    size=int(cfg.get("size", 256)),
                    clamp_recon=bool(exp.get("clamp_recon", False)),
                    adaptor=adaptor,
                    lambda_latent=float(exp.get("lambda_latent", 0.0)),
                    disc=disc, disc_cfg=cfg.get("discriminator"),
                )
                for k, v in terms.items():
                    totals.setdefault(k, []).append(v)
    finally:
        for m in moved:
            m.to("cpu")
        if moved and device.type == "cuda":
            torch.cuda.empty_cache()
    if was_training:
        dit.train()
    if adaptor is not None and was_training_ad:
        adaptor.train()
    if disc is not None and was_training_ds:
        disc.train()
    return {f"val_{k}": float(np.mean(v)) for k, v in totals.items()}


def resolve_dtd_root(noise_cfg: Dict[str, Any]) -> Path:
    """noise.dtd_root, else $DTD_ROOT, else latent_draem's download folder."""
    import os
    root = noise_cfg.get("dtd_root") or os.environ.get("DTD_ROOT")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[2] / "latent_draem" / "data" / "dtd"


def train(
    cfg: Dict[str, Any],
    rae_root: Path,
    mvtec_root: Path,
    out_dir: Path,
    device: torch.device,
    categories: Sequence[str],
    seed: int = 0,
    scope: str = "joint",
    ckpt_dir: Optional[Path] = None,
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
    sdpa = str(tcfg.get("sdpa", "auto")).lower()
    if sdpa not in ("auto", "math"):
        raise ValueError(f"training.sdpa must be 'auto' or 'math' (got '{sdpa}')")
    if sdpa == "math" and torch.cuda.is_available():
        # rule out the fused attention kernels as a NaN source (decoder + DiT)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        print("[iad] SDPA: math backend only (flash / memory-efficient attention off)",
              flush=True)
    loss_type = exp["loss"]
    corruption = exp.get("corruption", "simplenet")  # project default
    lambda_fm = float(exp.get("lambda_fm", 0.0))
    if loss_type == "hybrid" and corruption != "flow":
        raise ValueError("hybrid loss requires corruption='flow'")
    if corruption == "simplenet":
        if loss_type not in ("image_l1", "latent"):
            raise ValueError("corruption='simplenet' needs loss='image_l1' or 'latent' "
                             "(plus optional experiment.lambda_latent for image_l1)")
        core.simplenet_cfg(cfg["noise"])  # validate the noise block before any work
    if DS.disc_cfg(cfg.get("discriminator"))["enabled"] and corruption != "simplenet":
        raise ValueError("discriminator.enabled needs corruption='simplenet' "
                         "(its targets are the synthetic-anomaly masks)")
    if loss_type == "latent" and exp.get("decoder_finetune"):
        raise ValueError("loss='latent' never calls the decoder: decoder_finetune must be false")
    best_metric = exp.get("best_metric") or {"flow_matching": "fm",
                                             "latent": "lat"}.get(loss_type, "img")

    seed_all(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if corruption == "simplenet" and core.needs_textures(cfg["noise"]):
        # DRAEM texture pastes (noise.kind dtd / mix): preload a crop bank once
        dtd_root = resolve_dtd_root(cfg["noise"])
        n_tex = core.simplenet_cfg(cfg["noise"])["dtd_bank"]
        bank = core.build_texture_bank(dtd_root, n_tex, int(cfg.get("size", 256)), seed)
        core.set_texture_bank(bank)
        print(f"[iad] DTD texture bank: {bank.shape[0]} crops from {dtd_root}", flush=True)
    else:
        core.set_texture_bank(None)

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
    adaptor = A.build_adaptor(cfg.get("adaptor"),
                              dim=M.latent_dim_from_misc(misc, cfg), device=device)
    init_adaptor_state = copy.deepcopy(adaptor.state_dict()) if adaptor is not None else None
    if adaptor is not None:
        print(f"[iad] feature adaptor: {adaptor} "
              f"({M.count_params(adaptor)} params)", flush=True)
    transport = M.build_transport(M._to_dict(transport_cfg), misc)
    disc = DS.build_discriminator(cfg.get("discriminator"),
                                  dim=M.latent_dim_from_misc(misc, cfg), device=device, seed=seed)
    init_disc_state = copy.deepcopy(disc.state_dict()) if disc is not None else None
    if disc is not None:
        print(f"[iad] discriminator (DRAEM segmentation head, token grid): "
              f"{M.count_params(disc)} params, cfg={DS.disc_cfg(cfg.get('discriminator'))}",
              flush=True)

    scopes = [(c, [c]) for c in categories] if scope == "per_category" else [("joint", list(categories))]
    summaries = []
    for scope_name, cats in scopes:
        if scope == "per_category":
            # fresh DiT initialization + pristine decoder for every category
            dit.load_state_dict(init_state)
            rae.decoder.load_state_dict(init_decoder_state)
            if adaptor is not None:
                adaptor.load_state_dict(init_adaptor_state)
            if disc is not None:
                disc.load_state_dict(init_disc_state)
            if exp.get("decoder_finetune"):
                M.unfreeze_decoder(rae)
            else:
                M.freeze_decoder(rae)
        elif not exp.get("decoder_finetune"):
            M.freeze_decoder(rae)

        s = _train_scope(
            cfg=cfg, tcfg=tcfg, cats=cats, scope_name=scope_name, seed=seed,
            rae=rae, dit=dit, transport=transport, adaptor=adaptor,
            rae_root=rae_root, mvtec_root=Path(mvtec_root), out_dir=out_dir,
            ckpt_dir=ckpt_dir,
            device=device, misc=misc, num_classes=num_classes,
            n_steps=n_steps, bs=bs, precision=precision, clip_grad=clip_grad,
            log_every=log_every, val_every=val_every, save_every=save_every,
            n_val=n_val, size=size, use_ema=use_ema, ema_decay=ema_decay,
            cache=cache, loss_type=loss_type, corruption=corruption,
            lambda_fm=lambda_fm, best_metric=best_metric, resume=resume,
            build_scheduler=build_scheduler, disc=disc,
            # per category: eval-only best checkpoints, no periodic resume points
            slim=bool(tcfg.get("slim_checkpoints", scope == "per_category")),
        )
        summaries.append(s)
    return {"runs": summaries}


def _train_scope(
    *, cfg, tcfg, cats, scope_name, seed, rae, dit, transport,
    rae_root, mvtec_root, out_dir, device, misc, num_classes, ckpt_dir=None,
    n_steps, bs, precision, clip_grad, log_every, val_every, save_every,
    n_val, size, use_ema, ema_decay, cache, loss_type, corruption, lambda_fm,
    best_metric, resume, build_scheduler, adaptor=None, disc=None, slim=False,
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
    if cache and device.type == "cuda" and not (
            corruption == "simplenet" and core.needs_textures(cfg["noise"])):
        # exact: with cached latents the frozen encoder is never called again
        # (texture pastes re-encode every step, so the encoder stays then)
        M.offload_encoder(rae)
        print(f"[iad/{tag}] encoder offloaded to CPU (latents cached)", flush=True)
    if loss_type == "latent" and not exp.get("decoder_finetune") and device.type == "cuda":
        # exact: the decoder-free objective never calls the decoder
        rae.decoder.to("cpu")
        torch.cuda.empty_cache()
        print(f"[iad/{tag}] decoder offloaded to CPU (loss=latent never decodes)", flush=True)

    if adaptor is not None and adaptor.wants_stats:
        if latents is None:
            raise ValueError(
                "adaptor.init='train_stats' needs the train-split latents; "
                "set training.cache_latents: true")
        A.fit_train_stats(adaptor, latents, train_paths)
        print(f"[iad/{tag}] adaptor statistics fitted on {len(train_paths)} train "
              f"latents (init={adaptor.init}); "
              f"mean|z|={float(adaptor.train_mean.abs().mean()):.3f} "
              f"std range=[{float(adaptor.train_std.min()):.3f}, "
              f"{float(adaptor.train_std.max()):.3f}]", flush=True)

    # ---- optimizer / scheduler / EMA ----------------------------------------
    optimizer, base_lr, dec_lr, opt_msg = M.build_optimizer(
        dit, rae, bool(exp.get("decoder_finetune")), tcfg, adaptor=adaptor, disc=disc)
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

    ema_ad = None
    if adaptor is not None and use_ema:
        ema_ad = copy.deepcopy(adaptor)
        ema_ad.requires_grad_(False)
        ema_ad.eval()

    ema_ds = None
    if disc is not None and use_ema:  # paired with the EMA DiT at validation/eval
        ema_ds = copy.deepcopy(disc)
        ema_ds.requires_grad_(False)
        ema_ds.eval()

    start_step = 0
    if resume is not None:
        state = load_checkpoint(Path(resume))
        start_step = apply_checkpoint(state, dit, ema, rae, optimizer, scheduler,
                                      adaptor=adaptor, ema_adaptor=ema_ad,
                    disc=disc, ema_disc=ema_ds)
        print(f"[iad/{tag}] resumed from {resume} at step {start_step}", flush=True)

    autocast_ctx, scaler = _autocast(device, precision)
    trainable = list(dit.parameters())
    if adaptor is not None:
        trainable = trainable + list(adaptor.parameters())
    if exp.get("decoder_finetune"):
        trainable = trainable + list(rae.decoder.parameters())

    if ema is not None and str(tcfg.get("ema_device", "cuda")).lower() == "cpu" \
            and device.type == "cuda":
        # identical EMA math, only the storage device changes (frees ~4 bytes/param)
        ema.to("cpu")
        if ema_ad is not None:
            ema_ad.to("cpu")
        if ema_ds is not None:
            ema_ds.to("cpu")
        print(f"[iad/{tag}] EMA kept on CPU ({M.count_params(ema) / 1e6:.0f}M params "
              f"off the GPU)", flush=True)

    if bool(tcfg.get("grad_checkpoint", False)):
        n_ckpt = M.enable_gradient_checkpointing(dit, rae)
        print(f"[iad/{tag}] gradient checkpointing on {n_ckpt} DiT blocks"
              f"{' + decoder' if exp.get('decoder_finetune') else ''} "
              f"(recompute; no numerical change)", flush=True)

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
        "grad_checkpoint": bool(tcfg.get("grad_checkpoint", False)),
        "ema_device": str(tcfg.get("ema_device", "cuda")).lower(),
        "adaptor": A.adaptor_meta(adaptor),
        "discriminator": DS.disc_meta(disc, cfg.get("discriminator")),
        "size": size,
        "init": cfg.get("init", {}),
        "null_label": num_classes,
        "encoder": "stage1.RAE (frozen; weights referenced by rae_config)",
    }

    def ckpt_paths():
        prefix = tag if tag != "joint" else "joint"
        d = Path(ckpt_dir) if ckpt_dir is not None else out_dir
        return d / f"{prefix}_last.pt", d / f"{prefix}_best.pt"

    last_path, best_path = ckpt_paths()
    metrics_f = out_dir / "metrics.jsonl"
    best_val = math.inf
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)  # batch indices
    running: Dict[str, List[float]] = {}
    t_last = time.time()
    ema_model = ema if ema is not None else dit
    ad_model = ema_ad if ema_ad is not None else adaptor
    ds_model = ema_ds if ema_ds is not None else disc

    n_skip_dit = n_skip_head = 0              # non-finite-gradient guard
    max_skips = max(10, int(0.05 * n_steps))  # abort when it stops being rare
    step = start_step
    while step < n_steps:
        dit.train()
        if adaptor is not None:
            adaptor.train()
        if disc is not None:
            disc.train()
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
                    adaptor=adaptor,
                    lambda_latent=float(exp.get("lambda_latent", 0.0)),
                    disc=disc, disc_cfg=cfg.get("discriminator"),
                )
            # gradient accumulation: divide so the accumulated gradient is the
            # mean over the effective batch
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total / accum).backward()
            else:
                (total / accum).backward()
            for k, v in terms.items():
                micro_terms.setdefault(k, []).append(v)

        use_scaler = scaler is not None and scaler.is_enabled()
        if use_scaler:
            scaler.unscale_(optimizer)  # grads must be unscaled before clipping
        max_norm = clip_grad if clip_grad else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm)
        disc_norm = (torch.nn.utils.clip_grad_norm_(list(disc.parameters()), max_norm)
                     if disc is not None else None)  # own clip: never rescales the DiT
        # Non-finite guard: clip_grad_norm_ turns one inf/NaN gradient into NaN for
        # every element, and a single such update writes NaN into all weights for
        # good (seen on the image-L1 arms, job 152307). Skip that update instead.
        dit_ok = bool(torch.isfinite(grad_norm))
        head_ok = disc_norm is None or bool(torch.isfinite(disc_norm))
        if not head_ok:  # skip only the head's update (AdamW leaves grad=None params alone)
            for p in disc.parameters():
                p.grad = None
        if dit_ok:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.update()
        if not (dit_ok and head_ok):
            n_skip_dit += int(not dit_ok)
            n_skip_head += int(not head_ok)
            if n_skip_dit + n_skip_head <= 10:  # diagnose: forward (loss) or backward (grad)?
                step_terms = {k: [round(float(v), 6) for v in vals]
                              for k, vals in micro_terms.items()}
                print(f"[iad/{tag}] step {step + 1}: NON-FINITE gradient in "
                      f"{'the DiT/adaptor/decoder' if not dit_ok else 'the discriminator head'}"
                      f" (dit_norm={float(grad_norm):.4g}, head_norm="
                      f"{'n/a' if disc_norm is None else f'{float(disc_norm):.4g}'}); "
                      f"loss terms this step={step_terms}; update skipped", flush=True)
            if n_skip_dit > max_skips:
                raise RuntimeError(
                    f"{n_skip_dit} of {step + 1} steps had non-finite DiT gradients "
                    f"(limit {max_skips}): not a rare event any more, stopping")
        if scheduler is not None:
            scheduler.step()
        if dit_ok and (ema is not None or ema_ad is not None or ema_ds is not None):
            from utils.train_utils import update_ema  # official
            for em, mod in ((ema, dit), (ema_ad, adaptor), (ema_ds, disc)):
                if em is None:
                    continue
                with torch.no_grad():
                    if next(iter(em.parameters())).device != \
                            next(iter(mod.parameters())).device:
                        # CPU-resident EMA: same decay math, streamed from the GPU
                        for ep, p in zip(em.parameters(), mod.parameters()):
                            ep.mul_(ema_decay).add_(p.detach().to(ep.device),
                                                    alpha=1.0 - ema_decay)
                    else:
                        update_ema(em, mod, decay=ema_decay)

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
                "skipped_dit": n_skip_dit,
                "skipped_head": n_skip_head,
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
                            latents, cfg, device, bs, seed + 42, num_classes,
                            adaptor=ad_model, disc=ds_model)
            rec = {"step": step, "scope": tag, **vals}
            with metrics_f.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[iad/{tag}] step {step} " +
                  " ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
            key = f"val_{best_metric}"
            if key in vals and vals[key] < best_val:
                best_val = vals[key]
                save_checkpoint(best_path, step, dit, ema, rae, optimizer,
                                scheduler, cfg, meta,
                                adaptor=adaptor, ema_adaptor=ema_ad,
                                disc=disc, ema_disc=ema_ds, eval_only=slim)
                print(f"[iad/{tag}] new best {key}={best_val:.4f} -> {best_path}", flush=True)

        if save_every > 0 and step % save_every == 0 and not slim:
            # resume points only; eval-only (per-category) runs skip them: each
            # full 840M checkpoint write stalls training for minutes
            save_checkpoint(last_path, step, dit, ema, rae, optimizer,
                            scheduler, cfg, meta,
                            adaptor=adaptor, ema_adaptor=ema_ad,
                            disc=disc, ema_disc=ema_ds)

    if not slim:
        save_checkpoint(last_path, step, dit, ema, rae, optimizer, scheduler, cfg,
                        meta, adaptor=adaptor, ema_adaptor=ema_ad,
                        disc=disc, ema_disc=ema_ds)
    if best_val == math.inf and (val_paths or slim):
        save_checkpoint(best_path, step, dit, ema, rae, optimizer,
                        scheduler, cfg, meta,
                        adaptor=adaptor, ema_adaptor=ema_ad,
                        disc=disc, ema_disc=ema_ds, eval_only=slim)

    # explicit post-conditions (spec §6 / §2.3)
    enc_frozen = M.encoder_status(rae)
    print(f"[iad/{tag}] done at step {step}; encoder_frozen={enc_frozen}; "
          f"decoder_finetune={bool(exp.get('decoder_finetune'))}; "
          f"adaptor={getattr(adaptor, 'type', None)}; best_{best_metric}="
          f"{best_val if best_val is not math.inf else 'n/a'}", flush=True)

    if slim and best_path.exists():
        # per category: only the evaluated checkpoint, EMA weights only
        if last_path.exists() and best_path != last_path:
            last_path.unlink()   # a stale resume point from an older run
        print(f"[iad/{tag}] eval-only checkpoint {best_path.name}: "
              f"{best_path.stat().st_size / 1e9:.1f} GB (not resumable)", flush=True)

    if n_skip_dit or n_skip_head:
        print(f"[iad/{tag}] non-finite gradients: {n_skip_dit} DiT updates and "
              f"{n_skip_head} head updates skipped of {step} steps", flush=True)
    del images, latents
    return {
        "scope": tag,
        "steps": step,
        "best_val": None if best_val == math.inf else best_val,
        "skipped_dit_steps": n_skip_dit,
        "skipped_head_steps": n_skip_head,
        "encoder_frozen": enc_frozen,
        "last_ckpt": str(last_path),
        "best_ckpt": str(best_path) if best_path.exists() else None,
    }
