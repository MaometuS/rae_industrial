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
linear flow path ``z_t = (1-t) z_0 + t eps``).  Seeding: noise for image
index ``i``, repetition ``r`` is drawn from a generator seeded with
``seed -> (i, r)``, so results do not depend on batch size and are identical
across model variants given the same seed (spec §9.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iad.core import corrupt_latent, conditioning_time, prediction_to_clean_latent, warp_time


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
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Reconstruct a batch of images.

    Returns ``(recon_mean, maps, meta)`` where:

    * ``recon_mean``: (B,3,H,W) mean over repetitions of ``x_hat``;
    * ``maps``: (B,H,W) mean over repetitions of the *channel-averaged*
      absolute reconstruction error ``|x - x_hat|_mean-over-C`` (the pixel
      anomaly map before blurring);
    * ``meta``: forward-pass accounting for cost reporting.
    """
    assert mode in ("single", "multi", "ode"), f"unknown inference mode {mode}"
    assert corruption == "flow" or mode != "ode", \
        "ode integration requires corruption='flow' (linear path is the ODE the DiT was trained on)"
    reps = int(repeats) if mode == "multi" else 1
    reps = max(reps, 1)

    dit.eval()
    rae.eval()
    z0 = rae.encode(x)
    y = torch.full((x.shape[0],), int(num_classes), device=x.device, dtype=torch.long)
    tc = conditioning_time(torch.tensor([t], device=x.device), corruption).item()

    recon_sum = torch.zeros_like(x)
    map_sum = torch.zeros(x.shape[0], *x.shape[-2:], device=x.device, dtype=torch.float32)
    dit_passes = 0
    for rep in range(reps):
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
        xhat = rae.decode(zhat)
        if xhat.shape[-2:] != x.shape[-2:]:
            xhat = F.interpolate(xhat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        xhat = xhat.float().clamp(0.0, 1.0)
        recon_sum += xhat
        map_sum += (x - xhat).abs().mean(dim=1)

    meta = {
        "mode": mode,
        "t": t,
        "reps": reps,
        "dit_passes": dit_passes,
        "decode_passes": reps,
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
