"""Model construction and freezing helpers.

Everything model-related is delegated to the official RAE implementation in
``RAE_ROOT`` (default ``../rae_fidelity/RAE``):

* ``stage1.RAE`` — frozen DINOv2-B encoder + ViT-XL decoder (latent 768x16x16
  at 256 px, z-score normalized by ``models/stats/.../stat.pt``).
* ``stage2.models.DDT.DiTwDDTHead`` — the original RAE DiT (DiT-DH) loaded
  from the released checkpoint via ``instantiate_from_config`` (EMA weights,
  strict=True, exactly like ``src/sample.py``).
* ``stage2.transport.create_transport`` — the official flow-matching object.

This module only orchestrates them; no architecture code is duplicated.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_RAE_ROOT = Path(__file__).resolve().parent.parent.parent / "rae_fidelity" / "RAE"


# ---------------------------------------------------------------------------
# official-repo plumbing
# ---------------------------------------------------------------------------

@contextmanager
def pushd(path: Path):
    """Run a block with CWD ``path`` (official configs use repo-relative ckpt paths)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def setup_rae_root(rae_root: Path) -> Path:
    """Insert ``RAE_ROOT/src`` on sys.path (same approach as rae_fidelity.py)."""
    rae_root = Path(rae_root).resolve()
    src = rae_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(
            f"Official RAE repo not found at {rae_root} (expected {src}). "
            "Clone it: git clone https://github.com/bytetriper/RAE.git rae_fidelity/RAE"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return rae_root


_SECTION_NAMES = ("stage_1", "stage_2", "transport", "sampler",
                  "guidance", "misc", "training", "eval")


def load_official_sections(rae_config_path: Path, rae_root: Path,
                           overrides: Optional[Dict[str, Any]] = None):
    """Parse an official YAML into the 8 sections returned by ``parse_configs``.

    ``overrides`` optionally patches sections by name, e.g.
    ``{"stage_2": {"ckpt": "models/DiTs/.../stage2_model.pt"}}`` — used to add
    the released DiT checkpoint path to the official *training* config (which
    is the one carrying the original transport/optimizer settings).
    """
    from utils.train_utils import parse_configs  # official, requires setup_rae_root first

    rae_config_path = Path(rae_config_path)
    if not rae_config_path.is_absolute():
        rae_config_path = rae_root / rae_config_path
    with pushd(rae_root):
        sections = parse_configs(str(rae_config_path))
    sections = list(sections)  # (stage_1, stage_2, transport, sampler, guidance, misc, training, eval)
    if overrides:
        from omegaconf import OmegaConf
        for i, name in enumerate(_SECTION_NAMES):
            if name in overrides and sections[i] is not None:
                sections[i] = OmegaConf.merge(sections[i], OmegaConf.create(overrides[name]))
    return tuple(sections)


def _to_dict(cfg) -> Dict[str, Any]:
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_rae(rae_root: Path, rae_cfg: Dict[str, Any], device: torch.device,
              require_decoder_ckpt: bool = True,
              grid: Optional[int] = None) -> nn.Module:
    """Instantiate ``stage1.RAE`` exactly like ``stage1_sample.py``.

    If the decoder checkpoint is missing (unit-test environments) and
    ``require_decoder_ckpt`` is False, the decoder is randomly initialized —
    shapes and gradient paths are identical, so plumbing tests still work.
    """
    from utils.model_utils import instantiate_from_config

    rae_cfg = dict(rae_cfg)
    params = dict(rae_cfg.get("params", {}))
    decoder_path = params.get("pretrained_decoder_path")
    if decoder_path is not None:
        ckpt_abs = Path(decoder_path)
        if not ckpt_abs.is_absolute():
            ckpt_abs = rae_root / decoder_path
        if not ckpt_abs.is_file():
            if require_decoder_ckpt:
                raise FileNotFoundError(
                    f"RAE decoder checkpoint missing: {ckpt_abs}. "
                    "Download weights: (cd RAE_ROOT && hf download nyu-visionx/RAE-collections --local-dir models)"
                )
            print(f"[iad] WARNING: decoder checkpoint {ckpt_abs} missing -> random decoder (test mode)")
            params["pretrained_decoder_path"] = None
    rae_cfg["params"] = params

    with pushd(rae_root):
        rae = instantiate_from_config(rae_cfg)
    rae = rae.to(device)
    rae.eval()
    rae._dec_grid = int(round(rae.base_patches ** 0.5))  # the decoder's token grid (16)
    if grid is not None and int(grid) * rae.encoder_patch_size != rae.encoder_input_size:
        set_encoder_grid(rae, int(grid) * rae.encoder_patch_size)  # 32 -> 448 px
    return rae


def latent_grid(rae: nn.Module) -> int:
    """Side of the encoder's token grid (16 at the official 224 px input)."""
    return int(rae.encoder_input_size // rae.encoder_patch_size)


def _regrid_stat(t, grid: int):
    """Resize a per-position latent statistic ``(C, h, w)`` / ``(1, C, h, w)`` to
    ``grid x grid`` (bilinear); per-channel or scalar statistics are returned as is."""
    if not torch.is_tensor(t) or t.dim() < 3 or tuple(t.shape[-2:]) in ((1, 1), (grid, grid)):
        return t
    x = t if t.dim() == 4 else t.unsqueeze(0)
    y = F.interpolate(x.float(), size=(grid, grid), mode="bilinear", align_corners=False)
    return (y if t.dim() == 4 else y[0]).to(t.dtype)


def set_encoder_grid(rae: nn.Module, encoder_input_size: int) -> None:
    """Encode at ``encoder_input_size`` px (448 -> a 32x32 token grid) with the same
    frozen DINOv2 (HF interpolates its position embeddings).  The decoder keeps its
    16x16 grid: ``decode_latent`` pools a finer latent back before decoding.
    Per-position normalization statistics are resized to the new grid; the
    original ones are kept for decoding."""
    if int(encoder_input_size) % int(rae.encoder_patch_size):
        raise ValueError(f"encoder_input_size {encoder_input_size} must be a multiple of the "
                         f"encoder patch size {rae.encoder_patch_size}")
    rae.encoder_input_size = int(encoder_input_size)
    g = latent_grid(rae)
    if getattr(rae, "do_normalization", False):
        rae._dec_stats = (rae.latent_mean, rae.latent_var)
        rae.latent_mean = _regrid_stat(rae.latent_mean, g)
        rae.latent_var = _regrid_stat(rae.latent_var, g)
    print(f"[iad] encoder input {encoder_input_size} px -> {g}x{g} latent grid "
          f"(decoder stays {rae._dec_grid}x{rae._dec_grid})", flush=True)


def decode_latent(rae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """``rae.decode`` for any latent grid: a finer grid is average-pooled to the
    decoder's own grid (and de-normalized with the decoder-grid statistics)."""
    g = int(getattr(rae, "_dec_grid", z.shape[-1]))
    if z.shape[-1] == g:
        return rae.decode(z)
    z = F.adaptive_avg_pool2d(z, (g, g))
    stats = getattr(rae, "_dec_stats", None)
    if stats is None:
        return rae.decode(z)
    cur = (rae.latent_mean, rae.latent_var)
    rae.latent_mean, rae.latent_var = stats
    try:
        return rae.decode(z)
    finally:
        rae.latent_mean, rae.latent_var = cur


def regrid_dit(dit: nn.Module, grid: int) -> nn.Module:
    """Run a DiT-DH trained on a 16x16 grid on ``grid x grid`` tokens.

    Every weight is per-token (patch size 1), so only the positional parts change:
    the fixed sin-cos ``pos_embed`` is bicubically resized, the RoPE tables are
    rebuilt with positions interpolated onto the trained range (EVA's
    ``ft_seq_len``), and the patch embedders' expected input size is updated.
    Call it after loading the released 16x16 weights and before loading a
    checkpoint trained on ``grid``."""
    old = int(round(dit.s_embedder.num_patches ** 0.5))
    grid = int(grid)
    if grid == old:
        return dit
    from stage2.models.model_utils import VisionRotaryEmbeddingFast  # official
    for emb in (dit.s_embedder, dit.x_embedder):
        emb.img_size = (grid, grid)
        emb.grid_size = (grid, grid)
        emb.num_patches = grid * grid
    dev = next(dit.parameters()).device
    if getattr(dit, "use_pos_embed", False) and getattr(dit, "pos_embed", None) is not None:
        pe = dit.pos_embed.data
        c = pe.shape[-1]
        pe = pe.reshape(1, old, old, c).permute(0, 3, 1, 2).float()
        pe = F.interpolate(pe, size=(grid, grid), mode="bicubic", align_corners=False)
        dit.pos_embed = nn.Parameter(
            pe.permute(0, 2, 3, 1).reshape(1, grid * grid, c).to(dit.pos_embed.dtype),
            requires_grad=False)
    if getattr(dit, "use_rope", False):
        enc_half = dit.encoder_hidden_size // dit.num_heads[0] // 2
        dec_half = dit.decoder_hidden_size // dit.num_heads[1] // 2
        dit.enc_feat_rope = VisionRotaryEmbeddingFast(
            dim=enc_half, pt_seq_len=old, ft_seq_len=grid).to(dev)
        dit.dec_feat_rope = VisionRotaryEmbeddingFast(
            dim=dec_half, pt_seq_len=old, ft_seq_len=grid).to(dev)
    dit.iad_grid = grid
    print(f"[iad] DiT re-gridded {old}x{old} -> {grid}x{grid} (pos_embed resized, "
          f"RoPE interpolated)", flush=True)
    return dit


def build_dit(stage2_cfg: Dict[str, Any], rae_root: Path, device: torch.device,
              require_ckpt: bool = True, grid: Optional[int] = None) -> nn.Module:
    """Instantiate ``DiTwDDTHead`` with the official released checkpoint.

    The checkpoint (``stage_2.ckpt``) is handled by the official
    ``instantiate_from_config`` (loads the ``ema`` entry, strict=True).
    With ``require_ckpt=False`` the DiT is randomly initialized (unit tests).
    """
    from utils.model_utils import instantiate_from_config

    stage2_cfg = dict(stage2_cfg)
    if not require_ckpt:
        stage2_cfg.pop("ckpt", None)
    else:
        ckpt = stage2_cfg.get("ckpt")
        if ckpt is None:
            raise FileNotFoundError(
                "stage_2 config has no 'ckpt' key — set rae_config_overrides."
                "stage_2.ckpt in the experiment YAML (the official training "
                "config carries the original transport/optimizer settings but "
                "no checkpoint path), or pass init.from_pretrained: false.")
        ckpt_abs = Path(ckpt)
        if not ckpt_abs.is_absolute():
            ckpt_abs = rae_root / ckpt
        if not ckpt_abs.is_file():
            raise FileNotFoundError(
                f"DiT checkpoint missing: {ckpt_abs}. "
                "Download weights: (cd RAE_ROOT && hf download nyu-visionx/RAE-collections --local-dir models)"
            )

    with pushd(rae_root):
        dit = instantiate_from_config(stage2_cfg)
    dit = dit.to(device)
    dit.eval()  # dropout-free by default; .train() in the training loop
    if grid is not None:
        regrid_dit(dit, int(grid))
    return dit


def build_transport(transport_cfg: Dict[str, Any], misc_cfg: Dict[str, Any]):
    """Official ``create_transport`` with the official ``time_dist_shift``."""
    from stage2.transport import create_transport
    from iad.core import time_dist_shift_from_misc

    params = dict(transport_cfg.get("params", {}))
    params.pop("time_dist_shift", None)  # train.py pops this too
    return create_transport(
        **params,
        time_dist_shift=time_dist_shift_from_misc(misc_cfg),
    )


def latent_dim_from_misc(misc: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> int:
    """Latent channel count of the RAE (768 for DINOv2-B at 256 px).

    ``cfg`` may pin it explicitly (``latent_dim``); otherwise it is read from
    the official ``misc.latent_size`` so the feature adaptor always matches the
    encoder output.
    """
    if cfg is not None and cfg.get("latent_dim"):
        return int(cfg["latent_dim"])
    latent_size = misc.get("latent_size") if misc else None
    return int(latent_size[0]) if latent_size else 768


def null_labels(batch: int, num_classes: int, device: torch.device) -> torch.Tensor:
    """Class-conditional-free labels: the official CFG null token (1000).

    MVTec AD has no ImageNet label; official ``sample.py`` uses the null label
    for the unconditional branch, and ``LabelEmbedder`` maps it to the learned
    null embedding.  During training the official label dropout
    (``class_dropout_prob``) maps labels to the same null id, so passing null
    labels is a no-op for dropout.
    """
    return torch.full((batch,), int(num_classes), device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# freezing / parameter groups
# ---------------------------------------------------------------------------

def freeze_encoder(rae: nn.Module) -> None:
    """Encoder stays frozen in *all* experiments (spec §2.3)."""
    for p in rae.encoder.parameters():
        p.requires_grad_(False)
    # buffers (latent stats) are never optimized anyway
    rae.eval()


def freeze_decoder(rae: nn.Module) -> None:
    """Freeze decoder params while keeping forward ops differentiable.

    Deliberately *not* done under ``no_grad``: gradients must still flow
    through the decoder into the DiT (spec §5.3).
    """
    for p in rae.decoder.parameters():
        p.requires_grad_(False)
    rae.decoder.eval()


def unfreeze_decoder(rae: nn.Module) -> None:
    for p in rae.decoder.parameters():
        p.requires_grad_(True)
    rae.decoder.train()


def encoder_status(rae: nn.Module) -> bool:
    """True if every encoder parameter has requires_grad=False."""
    return all(not p.requires_grad for p in rae.encoder.parameters())


def offload_encoder(rae: nn.Module) -> None:
    """Move the frozen encoder to CPU once its latents are cached.

    Exactness: with ``cache_latents`` the encoder is never called again during
    training, so removing it from the GPU only frees memory.  ``encode_cache``
    moves it back to the device if another scope needs to encode.
    """
    rae.encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enable_gradient_checkpointing(dit: nn.Module, rae: Optional[nn.Module] = None) -> int:
    """Recompute activations in backward instead of storing them.

    Wraps every DiT block (and the RAE decoder, when fine-tuning) in
    ``torch.utils.checkpoint`` with ``use_reentrant=False``.  This changes only
    memory/compute (≈30% slower step, much smaller activation footprint), not
    the math: the forward values are recomputed exactly.  Calls under
    ``torch.no_grad`` (validation) run the wrapped function directly.

    Idempotent.  Returns the number of wrapped DiT blocks.
    """
    from torch.utils.checkpoint import checkpoint

    # fail loudly here rather than mid-run if this torch build cannot do
    # keyword-argument (non-reentrant) checkpointing
    probe = torch.ones(1, requires_grad=True)
    try:
        checkpoint(lambda a, k=None: a * (k if k is not None else 1.0),
                   probe, k=torch.ones(1), use_reentrant=False).sum().backward()
    except Exception as exc:  # pragma: no cover - depends on the torch build
        raise RuntimeError(
            "torch.utils.checkpoint with kwargs (use_reentrant=False) is "
            f"unusable in this environment: {exc}") from exc

    def wrap(fn):
        if getattr(fn, "_ckpt_wrapped", False):
            return fn

        def wrapped(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return fn(*args, **kwargs)

        wrapped._ckpt_wrapped = True  # type: ignore[attr-defined]
        return wrapped

    n = 0
    for blk in getattr(dit, "blocks", []):
        blk.forward = wrap(blk.forward)
        n += 1
    if rae is not None and hasattr(rae, "decode"):
        rae.decode = wrap(rae.decode)
    return n


def decoder_grad_norm(rae: nn.Module) -> float:
    total = 0.0
    for p in rae.decoder.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm(2).item() ** 2)
    return total ** 0.5


def build_optimizer(
    dit: nn.Module,
    rae: Optional[nn.Module],
    decoder_finetune: bool,
    train_cfg: Dict[str, Any],
    adaptor: Optional[nn.Module] = None,
    disc: Optional[nn.Module] = None,
):
    """AdamW with the official hyperparameters (utils.optim_utils defaults).

    Returns ``(optimizer, base_lr, decoder_lr)``.  When fine-tuning, the
    decoder gets its own param group at ``decoder_lr_ratio * lr`` (default
    0.1).  Adaptor parameters (when present) are trained with the DiT's
    learning rate.  Official ``build_scheduler`` clobbers *all* groups' lr to
    base_lr, so group learning rates are re-applied after scheduler
    construction by the caller (see ``iad.train``).
    """
    from utils.optim_utils import build_optimizer as official_build_optimizer

    opt_cfg = dict(train_cfg.get("optimizer", {}))
    lr = float(opt_cfg.get("lr", 2.0e-4))
    adaptor_params = ([p for p in adaptor.parameters() if p.requires_grad]
                      if adaptor is not None else [])
    ad_msg = f" + adaptor({sum(p.numel() for p in adaptor_params)} params)" \
        if adaptor_params else ""
    if disc is not None:  # DRAEM head: DiT lr/schedule; Adam is per-parameter, so
        disc_params = [p for p in disc.parameters() if p.requires_grad]  # no coupling
        adaptor_params = adaptor_params + disc_params
        ad_msg += f" + discriminator({sum(p.numel() for p in disc_params)} params)"

    if decoder_finetune:
        assert rae is not None, "decoder_finetune requires a RAE"
        dit_params = [p for p in dit.parameters() if p.requires_grad] + adaptor_params
        dec_params = [p for p in rae.decoder.parameters() if p.requires_grad]
        # official builder creates a single-group AdamW; we build the two-group
        # version with identical hyperparameters (same defaults as optim_utils).
        betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        eps = float(opt_cfg.get("eps", 1e-8))
        ratio = float(train_cfg.get("decoder_lr_ratio", 0.1))
        optimizer = torch.optim.AdamW(
            [
                {"params": dit_params, "lr": lr},
                {"params": dec_params, "lr": lr * ratio},
            ],
            betas=betas, weight_decay=weight_decay, eps=eps,
        )
        msg = (f"Optimizer: AdamW(lr={lr}, decoder_lr={lr * ratio}, "
               f"betas={betas}, wd={weight_decay}, eps={eps}) "
               f"[DiT{ad_msg} + decoder groups]")
        return optimizer, lr, lr * ratio, msg

    params = [p for p in dit.parameters() if p.requires_grad] + adaptor_params
    optimizer, msg = official_build_optimizer(params, train_cfg)
    return optimizer, lr, None, f"{msg}{ad_msg}"


def fix_param_group_lrs(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    decoder_lr: Optional[float],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Re-apply per-group learning rates after ``build_scheduler``.

    Official ``build_scheduler`` forces *every* param group to ``base_lr``
    before constructing the ``LambdaLR``, which then captures ``base_lrs``
    from those values — this would silently erase the decoder group's reduced
    lr.  We restore both the live lrs and the scheduler's captured base lrs so
    the decoder stays at ``decoder_lr`` (0.1x DiT lr) for the whole schedule.
    """
    optimizer.param_groups[0]["lr"] = base_lr
    if decoder_lr is not None and len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = decoder_lr
    if scheduler is not None:
        scheduler.base_lrs = [g["lr"] for g in optimizer.param_groups]


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters()
               if (p.requires_grad or not trainable_only))
