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
              require_decoder_ckpt: bool = True) -> nn.Module:
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
    return rae


def build_dit(stage2_cfg: Dict[str, Any], rae_root: Path, device: torch.device,
              require_ckpt: bool = True) -> nn.Module:
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
):
    """AdamW with the official hyperparameters (utils.optim_utils defaults).

    Returns ``(optimizer, base_lr, decoder_lr)``.  When fine-tuning, the
    decoder gets its own param group at ``decoder_lr_ratio * lr`` (default
    0.1).  Official ``build_scheduler`` clobbers *all* groups' lr to base_lr,
    so group learning rates are re-applied after scheduler construction by the
    caller (see ``iad.train``).
    """
    from utils.optim_utils import build_optimizer as official_build_optimizer

    opt_cfg = dict(train_cfg.get("optimizer", {}))
    lr = float(opt_cfg.get("lr", 2.0e-4))

    if decoder_finetune:
        assert rae is not None, "decoder_finetune requires a RAE"
        dit_params = [p for p in dit.parameters() if p.requires_grad]
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
               f"betas={betas}, wd={weight_decay}, eps={eps}) [DiT + decoder groups]")
        return optimizer, lr, lr * ratio, msg

    params = [p for p in dit.parameters() if p.requires_grad]
    optimizer, msg = official_build_optimizer(params, train_cfg)
    return optimizer, lr, None, msg


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
