"""Core math for RAE + DiT industrial anomaly detection experiments.

All equations are derived from the official RAE implementation
(``bytetriper/RAE``, see ``src/stage2/transport/``) rather than from the paper:

* Linear flow path (``transport/path.py::ICPlan``)::

      z_t = (1 - t) * z_data + t * eps,      eps ~ N(0, I)
      v*  = d/dt z_t = eps - z_data          (``compute_ut``)

  with ``t = 0`` the data endpoint and ``t = 1`` the noise endpoint.
* The DiT predicts the velocity ``v`` of this path (``ModelType.VELOCITY``,
  config ``prediction: velocity``).
* Exact clean-latent conversion (velocity endpoint formula)::

      z_hat = z_t - t * v          (data endpoint,  t -> 0)
      eps_hat = z_t + (1 - t) * v  (noise endpoint, t -> 1)

  At ``v = v*`` both are algebraically exact (unit-tested in ``tests/test_iad.py``).

An optional *additive* corruption ``z_t = z_0 + sigma * eps`` is provided for
noise ablations; its conditioning time is the flow-equivalent noise fraction
``t_cond = sigma / (1 + sigma)`` (see README "Known limitations").
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# time / noise-level sampling
# ---------------------------------------------------------------------------

# Time-distribution names accepted in configs.
TIME_DISTS = ("logit-normal_0_1", "uniform", "range", "fixed")


def time_dist_shift_from_misc(misc: Dict[str, Any]) -> float:
    """Official warp strength: sqrt(prod(latent_size) / 4096) (train.py)."""
    latent_size = tuple(int(d) for d in misc.get("latent_size", (768, 16, 16)))
    shift_dim = int(misc.get("time_dist_shift_dim", math.prod(latent_size)))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    return math.sqrt(shift_dim / shift_base)


def warp_time(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Official time warp ``t <- shift*t / (1 + (shift-1)*t)`` (transport.py)."""
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_time(
    noise_cfg: Dict[str, Any],
    transport: Any,
    z0: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample conditioning times ``t`` and noise ``eps`` for a latent batch.

    Reuses the official ``Transport.sample`` for the official distributions
    (``logit-normal_0_1`` and ``uniform`` incl. the ``time_dist_shift`` warp;
    those draw from the global RNG, so reproducibility goes through
    ``seed_all``).  ``range`` / ``fixed`` are experiment-only distributions;
    they return the *actual* path time used for corruption and conditioning
    (warp applied only when ``noise_cfg['warp']`` is true, off by default so
    the configured value is literal).
    """
    kind = str(noise_cfg.get("time_dist", "logit-normal_0_1"))

    if kind in ("logit-normal_0_1", "uniform"):
        # Official path: transport.sample(x1) draws its own eps and warps t.
        t, eps, _ = transport.sample(z0)
        return t.to(z0), eps.to(z0)

    eps = torch.randn_like(z0) if generator is None else \
        torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)

    if kind == "fixed":
        t = torch.full((z0.shape[0],), float(noise_cfg.get("t", 0.5)),
                       device=z0.device, dtype=z0.dtype)
    elif kind == "range":
        lo = float(noise_cfg.get("t_min", 0.0))
        hi = float(noise_cfg.get("t_max", 1.0))
        u = torch.rand((z0.shape[0],), device=z0.device, dtype=z0.dtype, generator=generator)
        t = lo + (hi - lo) * u
    else:
        raise ValueError(f"Unknown noise time_dist '{kind}'; expected one of {TIME_DISTS}")

    if noise_cfg.get("warp", False):
        t = warp_time(t, float(noise_cfg.get("time_dist_shift", 1.0)))
    return t, eps


# ---------------------------------------------------------------------------
# corruption
# ---------------------------------------------------------------------------

def corrupt_latent(
    z0: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    mode: str = "flow",
) -> torch.Tensor:
    """Corrupt the clean latent.

    ``flow`` (default, original RAE parameterization):
        ``z_t = (1 - t) * z0 + t * eps``
    ``additive`` (noise ablation only):
        ``z_t = z0 + t * eps`` where ``t`` is the raw sigma ``sigma``.
    """
    tb = t.view(-1, *([1] * (z0.dim() - 1)))
    if mode == "flow":
        return (1.0 - tb) * z0 + tb * eps
    if mode == "additive":
        return z0 + tb * eps
    raise ValueError(f"Unknown corruption mode '{mode}'; expected 'flow' or 'additive'")


def conditioning_time(t: torch.Tensor, corruption: str) -> torch.Tensor:
    """Time actually fed to the DiT for a given corruption mode.

    ``flow``: the path time itself.  ``additive``: the flow-equivalent noise
    fraction ``sigma / (1 + sigma)`` (so ``h`` below stays meaningful).
    """
    if corruption == "flow":
        return t
    if corruption == "additive":
        return t / (1.0 + t)
    raise ValueError(f"Unknown corruption mode '{corruption}'")


def prediction_to_clean_latent(
    zt: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
    corruption: str = "flow",
) -> torch.Tensor:
    """Convert the DiT velocity output into a clean-latent estimate.

    ``flow``: data endpoint of the linear path, ``z_hat = z_t - t * v``
    (exact when ``v = eps - z0``; also a one-step Euler integration of the
    probability-flow ODE from ``t`` to ``0``).

    ``additive``: same formula with the flow-equivalent conditioning time;
    exactness is not claimed — this mode exists for noise ablations.
    """
    tc = conditioning_time(t, corruption).view(-1, *([1] * (zt.dim() - 1)))
    return zt - tc * velocity


def prediction_to_noise(
    zt: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Noise endpoint of the linear path: ``eps_hat = z_t + (1 - t) * v``."""
    tb = t.view(-1, *([1] * (zt.dim() - 1)))
    return zt + (1.0 - tb) * velocity


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Per-sample mean over all non-batch dims (official transport/utils.py)."""
    return x.mean(dim=list(range(1, x.dim())))


def flow_matching_loss(
    transport: Any,
    dit: torch.nn.Module,
    z0: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official flow-matching loss on a *shared* (t, eps) draw.

    Identical in distribution and formula to
    ``Transport.training_losses`` for ``ModelType.VELOCITY``:
    ``mean_flat((v - ut)**2).mean()``.

    Returns ``(loss, zt, ut, v)``.
    """
    _, zt, ut = transport.path_sampler.plan(t, eps, z0)
    v = dit(zt, t, y=y)
    loss = mean_flat((v - ut) ** 2).mean()
    return loss, zt, ut, v


def official_flow_matching_loss(
    transport: Any,
    dit: torch.nn.Module,
    z0: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Pure Experiment-A loss: call the untouched official implementation.

    ``Transport.training_losses(model, x1, model_kwargs)`` with its internal
    time/noise sampling — this is exactly ``train.py``'s training step loss.
    """
    terms = transport.training_losses(dit, z0, model_kwargs=dict(y=y))
    return terms["loss"].mean()


def image_reconstruction_loss(
    rae: torch.nn.Module,
    zhat: torch.Tensor,
    images: torch.Tensor,
    size: int,
    clamp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode ``zhat`` and compare to the input image with L1.

    Reduction: ``F.l1_loss`` default ``reduction='mean'`` = mean absolute
    error over all pixels and channels, averaged across the batch.

    The decoder is *not* wrapped in ``no_grad``: gradients flow through its
    operations into the DiT (decoder params themselves are frozen via
    ``requires_grad_(False)``, see ``iad.models.freeze_decoder``).

    Returns ``(loss, reconstruction)``.
    """
    recon = rae.decode(zhat)
    if recon.shape[-2:] != images.shape[-2:]:
        recon = F.interpolate(recon, size=images.shape[-2:], mode="bilinear", align_corners=False)
    if clamp:
        recon = recon.clamp(0.0, 1.0)
    loss = F.l1_loss(recon, images)
    return loss, recon


def compute_losses(
    rae: torch.nn.Module,
    dit: torch.nn.Module,
    transport: Any,
    images: torch.Tensor,
    z0: torch.Tensor,
    loss_type: str,
    noise_cfg: Dict[str, Any],
    lambda_fm: float,
    y: torch.Tensor,
    corruption: str = "flow",
    size: int = 256,
    clamp_recon: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    """Build the training objective for one batch.

    ``loss_type``: ``flow_matching`` | ``image_l1`` | ``hybrid``.

    For ``flow_matching`` the untouched official ``training_losses`` is used
    (it draws its own time/noise).  For ``image_l1`` and ``hybrid`` a single
    shared ``(t, eps)`` draw and a single DiT forward feed both terms, so the
    hybrid's flow term is algebraically identical to the official loss.

    Returns ``(total_loss, terms_dict, zt, zhat)``: ``terms_dict`` holds the
    detached scalar terms (``total``/``img``/``fm`` as applicable), ``zt`` is
    the corrupted latent of the shared draw, and ``zhat`` the clean-latent
    estimate (``None`` for the pure flow-matching branch).
    """
    terms: Dict[str, float] = {}

    if loss_type == "flow_matching":
        # Untouched official objective; it draws its own (t, eps) internally.
        # No clean-latent conversion is needed for training (see infer.py for
        # the Experiment-A ODE reconstruction procedure).
        fm = official_flow_matching_loss(transport, dit, z0, y)
        terms["fm"] = float(fm.detach())
        terms["total"] = terms["fm"]
        t, eps = sample_time(noise_cfg, transport, z0, generator)
        _, zt, _ = transport.path_sampler.plan(t, eps, z0)
        return fm, terms, zt, None

    # shared draw for image / hybrid
    t, eps = sample_time(noise_cfg, transport, z0, generator)
    zt = corrupt_latent(z0, eps, t, mode=corruption)
    tc = conditioning_time(t, corruption)
    v = dit(zt, tc, y=y)
    zhat = prediction_to_clean_latent(zt, v, t, corruption=corruption)

    loss_img = None
    if loss_type in ("image_l1", "hybrid"):
        loss_img, _ = image_reconstruction_loss(rae, zhat, images, size, clamp=clamp_recon)
        terms["img"] = float(loss_img.detach())

    if loss_type == "hybrid":
        # official VELOCITY target on the same draw (flow corruption only;
        # for additive corruption the linear-path target is not defined)
        if corruption != "flow":
            raise ValueError("hybrid loss requires corruption='flow' "
                             "(additive corruption has no linear-path velocity target)")
        _, _, ut = transport.path_sampler.plan(t, eps, z0)
        loss_fm = mean_flat((v - ut) ** 2).mean()
        terms["fm"] = float(loss_fm.detach())
        total = loss_img + float(lambda_fm) * loss_fm
    elif loss_type == "image_l1":
        total = loss_img
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    terms["total"] = float(total.detach())
    return total, terms, zt, zhat
