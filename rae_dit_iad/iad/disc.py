"""DRAEM's discriminative sub-network, in the RAE latent space (every SimpleNet arm).

DRAEM (Zavrtanik et al., ICCV 2021) trains two networks: a reconstructive one
(here: the DiT) and a discriminative *segmentation* network that sees the
corrupted input and the reconstruction together and predicts where the
synthetic anomaly was pasted (focal loss against the mask).  At test time its
output, not the raw residual, is the anomaly map: it learns which differences
are defects and which are ordinary reconstruction error.

Here the head works on the 16x16 token grid (same module as latent_draem):

    logits = D([z_in, z^, z_in - z^, 1 - cos(z_in, z^)])
    L_seg  = focal_bce(logits, m),   m = SimpleNet token mask (soft coverage)
    score  = sigmoid(logits) -> bilinear 256 px -> blur -> metrics  (--score disc)

``detach: true`` (default) feeds it detached copies of z_in and z^, has its own
gradient clipping, and is initialized from a forked RNG: the head learns from
the DiT but never changes how the DiT trains (unit test N), so adding it leaves
every arm comparison intact and only adds a learned score on top.  Its loss is
reported as ``seg`` and kept out of ``total``, so it cannot change which DiT
checkpoint is selected as ``best``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

#: ``discriminator:`` config keys and their defaults
DISC_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "d": 256,        # hidden channels (multiple of 8: GroupNorm(8, d))
    "blocks": 4,     # residual conv blocks
    "w_seg": 1.0,    # weight of the focal segmentation loss
    "gamma": 2.0,    # focal-loss focusing parameter (DRAEM)
    "detach": True,  # True: the head never changes the DiT's training
}


def disc_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve and validate a ``discriminator:`` block."""
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(DISC_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown discriminator keys {sorted(unknown)}; "
                         f"expected {sorted(DISC_DEFAULTS)}")
    out = dict(DISC_DEFAULTS)
    out.update(cfg)
    out["enabled"] = bool(out["enabled"])
    out["d"], out["blocks"] = int(out["d"]), int(out["blocks"])
    out["w_seg"], out["gamma"] = float(out["w_seg"]), float(out["gamma"])
    out["detach"] = bool(out["detach"])
    if out["d"] <= 0 or out["d"] % 8:
        raise ValueError(f"discriminator.d must be a positive multiple of 8 (got {out['d']})")
    if out["blocks"] < 0 or out["w_seg"] <= 0 or out["gamma"] < 0:
        raise ValueError(f"invalid discriminator config {out}")
    return out


def build_discriminator(
    cfg: Optional[Dict[str, Any]],
    dim: int,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Optional[nn.Module]:
    """Build the head (``None`` when disabled) without touching the global RNG."""
    dc = disc_cfg(cfg)
    if not dc["enabled"]:
        return None
    from latent_draem import Discriminator  # identical module to latent_draem's `disc`
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 1_000_003)
        head = Discriminator(int(dim), dc["d"], dc["blocks"])
    return head.to(device) if device is not None else head


def seg_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """DRAEM's focal loss against the (soft) token mask."""
    from latent_draem import focal_bce
    return focal_bce(logits.float(), target.float(), gamma=float(gamma))


def disc_meta(head: Optional[nn.Module], cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if head is None:
        return None
    return {**disc_cfg(cfg), "params": int(sum(p.numel() for p in head.parameters()))}
