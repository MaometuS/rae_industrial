"""Inference: latent corruption -> DiT -> clean latent -> image -> anomaly map.

Documented inference modes (spec §9.2) — always report them together with
their cost (number of DiT/decoder forward passes per image):

``single``
    One corruption at noise level ``t`` (fixed), one DiT pass, decode.
    With the velocity parameterization this is *exactly* one Euler step of the
    probability-flow ODE from ``t`` to ``0``: ``z_hat = z_t - t * v``.
    Cost: 1 DiT pass + 1 decode.

``multi``
    ``repeats`` independent corruptions at noise level ``t`` (per-image,
    index-seeded noise — independent of batch size), anomaly scores are the
    mean of the per-repetition pixel maps.  Cost: ``repeats`` x single.

``ode``
    Iterative flow integration from the corrupted latent at ``t`` down to
    ``t = 0`` with explicit Euler on the official-style warped time grid
    (``warp(u) = s*u / (1 + (s-1)*u)``, ``s = sqrt(prod(latent)/4096)`` — the
    same warp the official sampler applies to its grid).  Cost:
    ``ode_steps`` DiT passes + 1 decode.

All modes corrupt with the *training* corruption (default: the original
linear flow path ``z_t = (1-t) z_0 + t eps``).  Exception: the SimpleNet-style
arms (``corruption: simplenet``, every arm but A) add **no** noise at test time -- the real
defect plays the role of the training corruption; ``t`` is then the training
sigma and only sets the conditioning time ``sigma / (1 + sigma)``.  That pass is
deterministic, so ``multi`` collapses to one repetition and ``ode`` is refused.

Anomaly maps: ``score='pixel'`` (default) is ``mean_c |x - x^|`` after the
decoder; ``latent_cos`` / ``latent_l2`` compare the test latent ``z`` with the
DiT's restoration ``z^`` token by token, before any decoder, and upsample the
16x16 token map to the image size (decoder-free scoring, experiment J; usable
with any checkpoint).  ``disc`` is DRAEM's learned map: ``sigmoid`` of the
discriminative head applied to (input latent, restoration), upsampled the same
way (SimpleNet arms with a ``discriminator:`` block only).  Seeding: noise for image
index ``i``, repetition ``r`` is drawn from a generator seeded with
``seed -> (i, r)``, so results do not depend on batch size and are identical
across model variants given the same seed (spec §9.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iad.core import (LATENT_SCORES, conditioning_time, corrupt_latent, latent_anomaly_map,
                      prediction_to_clean_latent, warp_time)


# ---------------------------------------------------------------------------
# noise plumbing (batch-size-invariant)
# ---------------------------------------------------------------------------

def _eps_for(image_idx: int, shape: Tuple[int, ...], device: torch.device,
             seed: int, rep: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) * 1_000_003 + int(image_idx) * 97 + int(rep) * 13)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


def _stack_eps(indices: Sequence[int], shape: Tuple[int, ...], device: torch.device,
               seed: int, rep: int) -> torch.Tensor:
    return torch.stack([_eps_for(i, shape, device, seed, rep) for i in indices])


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def _inv_warp(y: float, shift: float) -> float:
    """Inverse of ``warp_time``: solve ``y = s*u / (1 + (s-1)u)`` for ``u``."""
    if shift == 1.0:
        return float(y)
    return float(y) / (shift - (shift - 1.0) * float(y))


@torch.no_grad()
def _ode_to_data(
    dit: nn.Module,
    zt: torch.Tensor,
    t_start: float,
    steps: int,
    y: torch.Tensor,
    shift: float,
) -> torch.Tensor:
    """Euler integration of ``dz/dt = v(z, t)`` from ``t_start`` to ``0``.

    The time grid is the *official* warp applied to a uniform grid in warped
    coordinates (``t_i = warp(u_i)``, ``u_i`` uniform from
    ``warp^{-1}(t_start)`` down to 0), so the first node is exactly
    ``t_start`` — consistent with how the official integrator warps its full
    ``[1, 0]`` grid (``transport/integrators.py``), whose endpoints ``0`` and
    ``1`` are fixed points of the warp.
    """
    u0 = _inv_warp(float(t_start), shift)
    grid = warp_time(
        torch.linspace(u0, 0.0, int(steps) + 1, device=zt.device),
        shift,
    )
    z = zt
    b = z.shape[0]
    for i in range(int(steps)):
        t_i = grid[i]
        t_b = torch.full((b,), float(t_i), device=z.device, dtype=z.dtype)
        v = dit(z, t_b, y=y)
        z = z + v * (grid[i + 1] - grid[i])
    return z


@torch.no_grad()
def reconstruct(
    rae: nn.Module,
    dit: nn.Module,
    x: torch.Tensor,
    indices: Sequence[int],
    *,
    mode: str = "single",
    t: float = 0.5,
    repeats: int = 1,
    ode_steps: int = 50,
    num_classes: int = 1000,
    corruption: str = "flow",
    seed: int = 0,
    time_dist_shift: float = 1.0,
    adaptor: Optional[nn.Module] = None,
    score: str = "pixel",
    disc: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Reconstruct a batch of images.

    Returns ``(recon_mean, maps, meta)`` where:

    * ``recon_mean``: (B,3,H,W) mean over repetitions of ``x_hat``;
    * ``maps``: (B,H,W) mean over repetitions of the *channel-averaged*
      absolute reconstruction error ``|x - x_hat|_mean-over-C`` (the pixel
      anomaly map before blurring);
    * ``meta``: forward-pass accounting for cost reporting.

    ``adaptor`` (optional): feature adaptor applied to the encoded latent
    *before* corruption — identical to the training convention, so the DiT
    generates in the adapted space and the decoder decodes it.  No inverse is
    needed (the decoder is either fine-tuned or an identity-initialized
    residual keeps it in-distribution).
    """
    assert mode in ("single", "multi", "ode"), f"unknown inference mode {mode}"
    assert score in ("pixel", "disc") + LATENT_SCORES, f"unknown anomaly score {score}"
    assert score != "disc" or disc is not None, "score='disc' needs a discriminator"
    assert corruption == "flow" or mode != "ode", \
        "ode integration requires corruption='flow' (linear path is the ODE the DiT was trained on)"
    if corruption == "simplenet":
        repeats = 1  # no test-time noise: every repetition would be identical
    reps = int(repeats) if mode == "multi" else 1
    reps = max(reps, 1)

    dit.eval()
    rae.eval()
    if adaptor is not None:
        adaptor.eval()
    if disc is not None:
        disc.eval()
    z0 = rae.encode(x)
    if adaptor is not None:
        z0 = adaptor(z0)
    y = torch.full((x.shape[0],), int(num_classes), device=x.device, dtype=torch.long)
    tc = conditioning_time(torch.tensor([t], device=x.device), corruption).item()

    recon_sum = torch.zeros_like(x)
    map_sum = torch.zeros(x.shape[0], *x.shape[-2:], device=x.device, dtype=torch.float32)
    dit_passes = 0
    for rep in range(reps):
        if corruption == "simplenet":
            zt = z0  # SimpleNet-style: the real defect is the only "corruption"
        else:
            eps = _stack_eps(indices, tuple(z0.shape[1:]), x.device, seed, rep)
            zt = corrupt_latent(z0, eps, torch.full((z0.shape[0],), float(t), device=x.device),
                                mode=corruption)
        if mode == "ode":
            zhat = _ode_to_data(dit, zt, t, ode_steps, y, time_dist_shift)
            dit_passes += int(ode_steps)
        else:
            v = dit(zt, torch.full((z0.shape[0],), float(tc), device=x.device), y=y)
            zhat = prediction_to_clean_latent(zt, v, torch.tensor([t], device=x.device),
                                              corruption=corruption)
            dit_passes += 1
        # anchored adaptor (K): map the restoration back to the decoder's space
        from iad.models import decode_latent  # any latent grid (32x32 pooled to 16)
        xhat = decode_latent(rae, adaptor.invert(zhat) if getattr(adaptor, "is_anchored", False)
                             else zhat)
        if xhat.shape[-2:] != x.shape[-2:]:
            xhat = F.interpolate(xhat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        xhat = xhat.float().clamp(0.0, 1.0)
        recon_sum += xhat
        if score == "pixel":
            map_sum += (x - xhat).abs().mean(dim=1)
        elif score == "disc":  # DRAEM: the learned segmentation map, not the residual
            prob = torch.sigmoid(disc(zt, zhat)).float()
            map_sum += F.interpolate(prob, size=tuple(x.shape[-2:]), mode="bilinear",
                                     align_corners=False)[:, 0]
        else:  # decoder-free: compare latents before the decoder
            map_sum += latent_anomaly_map(z0, zhat, score, tuple(x.shape[-2:]))

    meta = {
        "mode": mode,
        "t": t,
        "reps": reps,
        "dit_passes": dit_passes,
        "decode_passes": reps,
        "adaptor": getattr(adaptor, "type", None),
        "corruption": corruption,
        "score": score,
    }
    return recon_sum / reps, map_sum / reps, meta


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def blur_maps(maps: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur pixel maps (project convention: sigma=4 at 256 px)."""
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(maps.astype(np.float32), sigma=(0, sigma, sigma))
    return maps.astype(np.float32)


def image_score(maps: np.ndarray, how: str = "max") -> np.ndarray:
    """Image-level aggregation of a pixel anomaly map.

    ``mean`` | ``max`` | ``top1`` (mean of the top 1% pixels).  ``max``/``top1``
    match ``latent_draem.image_scores``; ``mean`` is added here (spec §9.1
    treats the aggregation as an evaluation parameter).
    """
    flat = maps.reshape(len(maps), -1)
    if how == "mean":
        return flat.mean(1)
    if how == "max":
        return flat.max(1)
    if how == "top1":
        k = max(1, int(round(flat.shape[1] * 0.01)))
        return np.sort(flat, axis=1)[:, -k:].mean(1)
    raise ValueError(f"unknown image score '{how}'")


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------

def save_viz(
    images: np.ndarray,      # (N, H, W, 3) uint8 inputs
    recons: np.ndarray,      # (N, H, W, 3) uint8 reconstructions
    maps: np.ndarray,        # (N, H, W) float anomaly maps
    masks: np.ndarray,       # (N, H, W) bool ground truth
    labels: np.ndarray,      # (N,) bool/int 0=normal 1=anomalous
    path: Path,
    n_normal: int = 2,
    map_vmax: Optional[float] = None,
) -> None:
    """Grid: ``input | reconstruction | anomaly map | GT mask``.

    Rows: first defective images, then normal ones (project convention from
    ``rae_recon.save_viz``).  Heat map scale defaults to 3x the 99th
    percentile of *normal* maps so defects are not hidden by outliers
    (``rae_fidelity`` convention).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image

    anomalous = [i for i in range(len(labels)) if labels[i]]
    normal = [i for i in range(len(labels)) if not labels[i]]
    rows = anomalous[: max(n_normal * 2, 4)] + normal[:n_normal]
    if not rows:
        rows = list(range(min(6, len(labels))))
    if map_vmax is None:
        normal_maps = maps[~labels.astype(bool)] if (~labels.astype(bool)).any() else maps
        map_vmax = max(3.0 * float(np.percentile(normal_maps, 99)), 1e-6)

    cells = []
    for i in rows:
        heat = cm.inferno(np.clip(maps[i] / map_vmax, 0, 1))[..., :3]
        gt = np.zeros_like(heat)
        gt[..., 0] = masks[i]  # red mask on black
        cells.append([
            images[i],
            recons[i],
            (heat * 255).astype(np.uint8),
            (gt * 255).astype(np.uint8),
        ])
    rows_img = [np.concatenate(row, axis=1) for row in cells]
    grid = np.concatenate(rows_img, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


# ---------------------------------------------------------------------------
# localization ablation: many score variants from one pass (SimpleNet arms)
# ---------------------------------------------------------------------------
#
# A score spec is ``base[+modifier...]``:
#
# base (one DiT pass yields all of them)
#   pixel        mean_c |x - D(z^)|
#   dec_diff     mean_c |D(z) - D(z^)|: the decoder's own error cancels
#   latent_cos / latent_l2 / disc   as in ``reconstruct``
#   pxhead       the pixel-resolution segmentation head (``discriminator.pixel``)
#   draem        DRAEM's U-Net on [x, D(z^)] (``discriminator.type: unet``, loss draem)
#   fuse_dec     latent_cos * dec_diff     (coarse "where" x sharper "what")
#   fuse_pixel   latent_cos * pixel
#   fuse_disc    disc * dec_diff
#   oracle16 / oracle32   the ground-truth mask area-pooled to a GxG grid and
#                bilinearly upsampled: the best any GxG token map can do
#                (computed from the masks in ``cmd_eval``, no model)
# modifiers
#   sK           shift TTA: K in {4, 16} sub-patch offsets of the input
#                (half / quarter patch), maps un-shifted and averaged
#   gf           guided-filter upsampling with the input image as guide (radius =
#                one token, eps 1e-3: the map's edges snap to the image's edges)

PASS_BASES = ("pixel", "dec_diff", "latent_cos", "latent_l2", "disc", "pxhead", "draem")
FUSED_BASES = {"fuse_dec": ("latent_cos", "dec_diff"),
               "fuse_pixel": ("latent_cos", "pixel"),
               "fuse_disc": ("disc", "dec_diff")}
ORACLE_BASES = ("oracle16", "oracle32")
SHIFT_SETS = (1, 4, 16)


def parse_score_spec(spec: str) -> Dict[str, Any]:
    """``"fuse_dec+s4+gf"`` -> ``{base, shifts, gf, needs}`` (validated)."""
    parts = [p for p in str(spec).strip().split("+") if p]
    if not parts:
        raise ValueError("empty score spec")
    base, mods = parts[0], parts[1:]
    if base not in PASS_BASES + tuple(FUSED_BASES) + ORACLE_BASES:
        raise ValueError(f"unknown score base '{base}' in '{spec}'")
    shifts, gf = 1, False
    for m in mods:
        if m == "gf":
            gf = True
        elif m.startswith("s") and m[1:].isdigit() and int(m[1:]) in SHIFT_SETS:
            shifts = int(m[1:])
        else:
            raise ValueError(f"unknown score modifier '{m}' in '{spec}' "
                             f"(sK with K in {SHIFT_SETS}, gf)")
    if base in ORACLE_BASES and shifts != 1:
        raise ValueError(f"'{spec}': shift TTA does not apply to an oracle")
    needs = FUSED_BASES.get(base, (base,)) if base not in ORACLE_BASES else ()
    return {"spec": str(spec), "base": base, "shifts": shifts, "gf": gf, "needs": needs}


def shift_offsets(k: int, patch_px: float) -> List[Tuple[int, int]]:
    """``k`` (dy, dx) input offsets in pixels on an even sub-patch lattice."""
    n = int(round(k ** 0.5))
    if n * n != k:
        raise ValueError(f"shift count {k} is not a square")
    steps = sorted({int(round(i * patch_px / n)) for i in range(n)})  # unique (tiny patches)
    return [(dy, dx) for dy in steps for dx in steps]


def shift_image(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """``out[i, j] = x[i + dy, j + dx]`` (reflect-padded at the bottom/right)."""
    if dy == 0 and dx == 0:
        return x
    return F.pad(x, (0, dx, 0, dy), mode="reflect")[..., dy:, dx:]


def unshift_map(m: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Inverse of ``shift_image`` for a ``(B, H, W)`` map (edge-replicated)."""
    if dy == 0 and dx == 0:
        return m
    h, w = m.shape[-2:]
    return F.pad(m.unsqueeze(1), (dx, 0, dy, 0), mode="replicate")[:, 0, :h, :w]


def guided_filter(p: torch.Tensor, guide: torch.Tensor, r: int, eps: float = 1e-3) -> torch.Tensor:
    """He et al.'s guided filter: edge-aware smoothing of ``p`` ``(B, H, W)``
    steered by the grey-level ``guide`` ``(B, 3, H, W)`` in [0, 1].  Linear in
    ``p``, so the map's scale does not matter; ``eps`` is relative to the guide."""
    def box(a):
        return F.avg_pool2d(a, 2 * r + 1, stride=1, padding=r, count_include_pad=False)
    g = guide.float().mean(dim=1, keepdim=True)
    q = p.float().unsqueeze(1)
    mg, mp = box(g), box(q)
    a = (box(g * q) - mg * mp) / (box(g * g) - mg * mg + eps)
    b = mp - a * mg
    return (box(a) * g + box(b))[:, 0]


def mask_oracle(masks: np.ndarray, grid: int) -> np.ndarray:
    """Ground-truth masks seen through a ``grid x grid`` token map: area-pooled,
    then bilinearly upsampled like every token score.  Ties among the zero
    pixels of normal images are broken by a fixed tiny noise."""
    m = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
    tok = F.adaptive_avg_pool2d(m, (grid, grid))
    up = F.interpolate(tok, size=masks.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    noise = np.random.default_rng(0).random(up.shape).astype(np.float32) * 1e-6
    return up.numpy() + noise


@torch.no_grad()
def score_maps(
    rae: nn.Module,
    dit: nn.Module,
    x: torch.Tensor,
    specs: Sequence[Dict[str, Any]],
    *,
    t: float,
    num_classes: int = 1000,
    adaptor: Optional[nn.Module] = None,
    disc: Optional[nn.Module] = None,
    patch_px: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Every requested score spec (``parse_score_spec``) for one batch of a
    SimpleNet-path checkpoint (deterministic single pass, no test noise).

    One DiT pass per shift offset; all bases needed by the specs come out of
    the same pass.  Shift sets are nested (s1 in s4 in s16), so s16 also
    yields s4 and s1.  Returns ``(x_hat of the unshifted pass, {spec: map})``,
    maps ``(B, H, W)`` at the input resolution, before blur."""
    from iad.models import decode_latent, latent_grid
    specs = [s for s in specs if s["base"] not in ORACLE_BASES]
    if not specs:
        return x.new_zeros(x.shape), {}
    bases = sorted({b for s in specs for b in s["needs"]})
    if ("disc" in bases or "pxhead" in bases) and disc is None:
        raise ValueError("disc / pxhead / fuse_disc scores need a discriminator")
    if "pxhead" in bases and getattr(disc, "pix", None) is None:
        raise ValueError("pxhead needs a checkpoint trained with discriminator.pixel: true")
    unet = bool(getattr(disc, "is_unet", False))
    if "draem" in bases and not unet:
        raise ValueError("draem needs a checkpoint trained with discriminator.type: unet")
    if unet and ({"disc", "pxhead"} & set(bases)):
        raise ValueError("disc / pxhead / fuse_disc need a token-grid head, not the U-Net")
    kmax = max(s["shifts"] for s in specs)
    if patch_px is None:  # input pixels per token
        g = (latent_grid(rae) if hasattr(rae, "encoder_input_size")
             else rae.encode(x[:1]).shape[-1])
        patch_px = x.shape[-1] / float(g)
    offsets = shift_offsets(kmax, patch_px)
    subsets = {k: set(shift_offsets(k, patch_px)) for k in SHIFT_SETS if k <= kmax}

    dit.eval()
    rae.eval()
    y = torch.full((x.shape[0],), int(num_classes), device=x.device, dtype=torch.long)
    tc = conditioning_time(torch.tensor([t], device=x.device), "simplenet").item()
    anchored = getattr(adaptor, "is_anchored", False)
    sums = {k: {b: 0.0 for b in bases} for k in subsets}
    xhat0 = None
    for dy, dx in offsets:
        xs = shift_image(x, dy, dx)
        z0 = rae.encode(xs)
        if adaptor is not None:
            z0 = adaptor(z0)
        v = dit(z0, torch.full((x.shape[0],), float(tc), device=x.device), y=y)
        zhat = prediction_to_clean_latent(z0, v, torch.tensor([t], device=x.device),
                                          corruption="simplenet")
        hw = tuple(xs.shape[-2:])
        got: Dict[str, torch.Tensor] = {}
        xh = None
        if {"pixel", "dec_diff", "draem"} & set(bases) or (dy, dx) == (0, 0):
            xh = decode_latent(rae, adaptor.invert(zhat) if anchored else zhat)
            if xh.shape[-2:] != hw:
                xh = F.interpolate(xh, size=hw, mode="bilinear", align_corners=False)
            xh = xh.float().clamp(0.0, 1.0)
            if (dy, dx) == (0, 0):
                xhat0 = xh
        if "pixel" in bases:
            got["pixel"] = (xs - xh).abs().mean(dim=1)
        if "dec_diff" in bases:
            xz = decode_latent(rae, adaptor.invert(z0) if anchored else z0)
            if xz.shape[-2:] != hw:
                xz = F.interpolate(xz, size=hw, mode="bilinear", align_corners=False)
            got["dec_diff"] = (xz.float().clamp(0.0, 1.0) - xh).abs().mean(dim=1)
        for b in ("latent_cos", "latent_l2"):
            if b in bases:
                got[b] = latent_anomaly_map(z0, zhat, b, hw)
        if "draem" in bases:
            got["draem"] = torch.sigmoid(disc(xs, xh.to(xs.dtype))).float()[:, 0]
        if "disc" in bases or "pxhead" in bases:
            logit = disc(z0, zhat)
            if "disc" in bases:
                got["disc"] = F.interpolate(torch.sigmoid(logit).float(), size=hw,
                                            mode="bilinear", align_corners=False)[:, 0]
            if "pxhead" in bases:
                got["pxhead"] = torch.sigmoid(disc.pixel(xs, z0, zhat, logit)).float()[:, 0]
        for b, m in got.items():
            m = unshift_map(m.float(), dy, dx)
            for k, sub in subsets.items():
                if (dy, dx) in sub:
                    sums[k][b] = sums[k][b] + m
    out: Dict[str, torch.Tensor] = {}
    for s in specs:
        k = s["shifts"]
        avg = {b: sums[k][b] / float(len(subsets[k])) for b in s["needs"]}
        if s["base"] in FUSED_BASES:
            a, b = FUSED_BASES[s["base"]]
            m = avg[a] * avg[b]
        else:
            m = avg[s["base"]]
        if s["gf"]:
            m = guided_filter(m, x, r=max(1, int(round(patch_px))))  # one token wide
        out[s["spec"]] = m
    return xhat0, out
