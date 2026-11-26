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

``pixel: true`` (localization ablation) adds a pixel-resolution head on top
(``PixelHead``): RGB stem at 1/4 resolution + the token cues (z_in, z^, cos
distance, the token logit) upsampled, two residual blocks, a full-resolution
refinement, and a zero-initialized output added to the upsampled token logit
(so at step 0 it IS the bilinear ``disc`` map).  It is trained with the same
focal loss against the *pixel* Perlin mask on the samples whose corruption is
visible in the image (DTD paste) and on clean samples (all-zero target);
Gaussian / swap samples change only the latent, so they carry no pixel target.
Its inputs are detached (the token head and the DiT train exactly as without
it) and it is clipped separately.  Score: ``--score pxhead``.

``type: unet`` (exact-DRAEM arm, ``loss: draem``) replaces the token head by
DRAEM's own discriminative sub-network (``DraemUNet``): a U-Net on the
concatenated [augmented image, reconstruction D(z^)] at image resolution,
focal loss against the pixel Perlin mask.  Score: ``--score draem``.
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
    "pixel": False,  # pixel-resolution head on top (localization ablation)
    "pixel_ch": 32,  # its width (multiple of 8)
    "w_pix": 1.0,    # weight of its focal loss
    "type": "token",  # token: head on the latent grid | unet: DRAEM's image-space U-Net
    "unet_base": 64,  # DRAEM's base width (multiple of 8)
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
    out["pixel"], out["pixel_ch"], out["w_pix"] = (bool(out["pixel"]), int(out["pixel_ch"]),
                                                   float(out["w_pix"]))
    out["type"], out["unet_base"] = str(out["type"]).lower(), int(out["unet_base"])
    if out["type"] not in ("token", "unet"):
        raise ValueError(f"discriminator.type must be token or unet (got {out['type']})")
    if out["type"] == "unet" and (out["pixel"] or out["unet_base"] <= 0 or out["unet_base"] % 8):
        raise ValueError("discriminator.type unet: pixel must be false and unet_base a "
                         f"positive multiple of 8 (got {out})")
    if out["pixel"] and (out["pixel_ch"] <= 0 or out["pixel_ch"] % 8 or out["w_pix"] <= 0):
        raise ValueError(f"invalid discriminator pixel head config {out}")
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
    if dc["type"] == "unet":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1_000_003)
            head = DraemUNet(base=dc["unet_base"])
        return head.to(device) if device is not None else head
    from latent_draem import Discriminator  # identical module to latent_draem's `disc`
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 1_000_003)
        head = Discriminator(int(dim), dc["d"], dc["blocks"])
        if dc["pixel"]:  # built after the token head: its init is unchanged
            head = DiscWithPixel(head, PixelHead(int(dim), dc["pixel_ch"]))
    return head.to(device) if device is not None else head


def is_unet(head: Optional[nn.Module]) -> bool:
    return bool(getattr(head, "is_unet", False))


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                         nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout),
                         nn.ReLU(inplace=True))


class DraemUNet(nn.Module):
    """DRAEM's ``DiscriminativeSubNetwork`` (Zavrtanik et al. 2021, official code):
    input [augmented image, reconstruction] (6 channels), six encoder stages
    (b, 2b, 4b, 8b, 8b, 8b; five 2x poolings), a skip-connected decoder back to
    full resolution, one output logit per pixel.

    Deviations: GroupNorm instead of BatchNorm (our micro-batch is 2, BatchNorm
    statistics at that size are noise) and one sigmoid logit instead of a
    2-way softmax (the same binary focal loss).  Input side must be divisible by 32."""

    is_unet = True

    def __init__(self, in_ch: int = 6, base: int = 64):
        super().__init__()
        b = int(base)
        self.e1, self.e2, self.e3 = _block(in_ch, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.e4, self.e5, self.e6 = _block(4 * b, 8 * b), _block(8 * b, 8 * b), _block(8 * b, 8 * b)
        self.pool = nn.MaxPool2d(2)
        self.u5, self.d5 = _up(8 * b, 8 * b), _block(16 * b, 8 * b)
        self.u4, self.d4 = _up(8 * b, 4 * b), _block(12 * b, 4 * b)
        self.u3, self.d3 = _up(4 * b, 2 * b), _block(6 * b, 2 * b)
        self.u2, self.d2 = _up(2 * b, b), _block(3 * b, b)
        self.u1, self.d1 = _up(b, b), _block(2 * b, b)
        self.out = nn.Conv2d(b, 1, 3, padding=1)

    def forward(self, x_in: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        h, w = x_in.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"DraemUNet needs a side divisible by 32 (got {h}x{w})")
        f1 = self.e1(torch.cat([x_in, x_rec], 1))
        f2 = self.e2(self.pool(f1))
        f3 = self.e3(self.pool(f2))
        f4 = self.e4(self.pool(f3))
        f5 = self.e5(self.pool(f4))
        f6 = self.e6(self.pool(f5))
        d = self.d5(torch.cat([self.u5(f6), f5], 1))
        d = self.d4(torch.cat([self.u4(d), f4], 1))
        d = self.d3(torch.cat([self.u3(d), f3], 1))
        d = self.d2(torch.cat([self.u2(d), f2], 1))
        d = self.d1(torch.cat([self.u1(d), f1], 1))
        return self.out(d)


class PixelHead(nn.Module):
    """Pixel-resolution anomaly logit from the image and the token-grid cues."""

    def __init__(self, dim: int, ch: int = 32):
        super().__init__()
        from latent_draem import ResBlock
        self.tok = nn.Conv2d(2 * dim + 2, ch, 1)          # [z_in, z^, cos, token logit]
        self.stem = nn.Sequential(                           # RGB -> 1/4 resolution
            nn.Conv2d(3, ch, 3, stride=2, padding=1), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        self.mid = nn.Sequential(nn.Conv2d(2 * ch, ch, 1), ResBlock(ch), ResBlock(ch))
        self.full = nn.Conv2d(3, ch // 2, 3, padding=1)      # full-resolution edges
        self.out = nn.Sequential(nn.Conv2d(ch + ch // 2, ch // 2, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(ch // 2, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, za, r, logit):
        import torch.nn.functional as F
        h, w = x.shape[-2:]
        cd = 1 - F.cosine_similarity(za, r, dim=1, eps=1e-6).unsqueeze(1)
        tok = self.tok(torch.cat([za, r, cd, logit], 1))
        s = self.stem(x)
        tok = F.interpolate(tok, size=s.shape[-2:], mode="bilinear", align_corners=False)
        f = self.mid(torch.cat([s, tok], 1))
        f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
        res = self.out(torch.cat([f, self.full(x)], 1))
        base = F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
        return base + res


class DiscWithPixel(nn.Module):
    """Token head (``seg``, unchanged) + pixel head (``pix``).  ``forward`` is the
    token head, so every existing caller is unaffected."""

    def __init__(self, seg: nn.Module, pix: nn.Module):
        super().__init__()
        self.seg, self.pix = seg, pix

    def forward(self, za, r):
        return self.seg(za, r)

    def pixel(self, x, za, r, logit):
        return self.pix(x, za.detach(), r.detach(), logit.detach())


def head_parts(head: Optional[nn.Module]):
    """Parameter groups clipped separately (token head, pixel head)."""
    if head is None:
        return []
    if isinstance(head, DiscWithPixel):
        return [list(head.seg.parameters()), list(head.pix.parameters())]
    return [list(head.parameters())]


def seg_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """DRAEM's focal loss against the (soft) token mask."""
    from latent_draem import focal_bce
    return focal_bce(logits.float(), target.float(), gamma=float(gamma))


def disc_meta(head: Optional[nn.Module], cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if head is None:
        return None
    return {**disc_cfg(cfg), "params": int(sum(p.numel() for p in head.parameters()))}


def clip_head(head: Optional[nn.Module], max_norm: float) -> Optional[torch.Tensor]:
    """Clip each head part separately; return their combined norm (CPU scalar).

    A part that got no gradient this step (the pixel head on a batch with no
    DTD / clean sample) reports a CPU zero while the others are on the GPU, so
    the norms are combined on the CPU (job 163739 failed on exactly that)."""
    norms = [torch.nn.utils.clip_grad_norm_(ps, max_norm) for ps in head_parts(head)]
    if not norms:
        return None
    return torch.stack([n.detach().float().cpu() for n in norms]).norm()
