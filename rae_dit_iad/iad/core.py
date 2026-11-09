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

*SimpleNet-style* corruption (the project default for every arm except A,
``corruption: simplenet``) adds
small i.i.d. Gaussian noise to a Perlin-shaped subset of tokens only::

    z_in = z_0 + m * sigma * eps,     m in [0, 1]^(B,1,h,w),  sigma small

and grades the DiT against the clean latent.  Unlike the flow path it leaves
the untouched tokens clean, so the model learns "replace tokens that do not
fit their (clean) context" rather than global denoising.  The conditioning
time is ``sigma / (1 + sigma)`` as for ``additive``; at test time no noise is
added (the real defect plays the role of the training corruption).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
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
    if corruption in ("additive", "simplenet"):
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


# ---------------------------------------------------------------------------
# SimpleNet-style corruption (default for every arm except A)
# ---------------------------------------------------------------------------

#: ``noise:`` keys read by ``corruption: simplenet`` and their defaults
SIMPLENET_DEFAULTS: Dict[str, Any] = {
    # i.i.d. Gaussian std in normalized-latent units (each channel ~unit
    # variance, so a token's squared norm is ~768 and sigma=0.1 adds ~1%).
    # SimpleNet uses sigma=0.015 in its own adapted WideResNet feature scale,
    # which does not transfer 1:1 -- sweep it (``--noise-sigma`` / ``SIGMA``).
    # A two-element list ``[lo, hi]`` draws sigma ~ U(lo, hi) per sample
    # (latent_draem's working range is [0.5, 1.5]); test time then conditions
    # on the midpoint.  The first full run showed 0.1 is far too weak: the
    # restorer learns a near-identity map and copies real defects through.
    "sigma": 0.1,
    # What goes into the Perlin region (DRAEM-style synthetic anomaly):
    #   gaussian  z0 + m*sigma*eps                       (SimpleNet noise; J, J2)
    #   swap      z0 + beta*m*(roll(z_j) - z0)           (misplaced normal tokens:
    #             another image of the batch, spatially shifted, maybe flipped)
    #   dtd       E(x + beta*M*(texture - x))             (DRAEM: a DTD texture
    #             pasted in pixel space, then encoded -- needs a texture bank)
    #   mix       one of ``kinds`` per sample
    # Same beta range and texture recipe as latent_draem (pixel / latent_self).
    "kind": "gaussian",
    "kinds": ["gaussian", "swap", "dtd"],
    "beta": [0.5, 1.0],
    "dtd_bank": 1000,   # random DTD crops preloaded once (fp16, CPU)
    "mask": "perlin",   # perlin (DRAEM-style token blobs) | all (SimpleNet: every token)
    "p_clean": 0.2,     # fraction of samples left uncorrupted (learn identity on normals)
    "w_anom": 5.0,      # latent-loss up-weight on corrupted tokens (lambda_latent > 0)
    "mask_thr": 0.5,    # Perlin threshold (latent_draem.perlin_mask convention)
}


CORRUPTION_KINDS = ("gaussian", "swap", "dtd")


def simplenet_cfg(noise_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve and validate the SimpleNet-style keys of a ``noise:`` block."""
    cfg = dict(SIMPLENET_DEFAULTS)
    for k in SIMPLENET_DEFAULTS:
        if noise_cfg and k in noise_cfg:
            cfg[k] = noise_cfg[k]
    sig = cfg["sigma"]
    if isinstance(sig, str):
        sig = [float(v) for v in sig.replace(",", " ").split()]
    if isinstance(sig, (list, tuple)):
        if len(sig) == 1:
            sig = sig[0]
        elif len(sig) != 2 or not float(sig[1]) >= float(sig[0]):
            raise ValueError(f"simplenet sigma range must be [lo, hi] with hi >= lo (got {sig})")
    if isinstance(sig, (list, tuple)):
        lo, hi = float(sig[0]), float(sig[1])
    else:
        lo = hi = float(sig)
    cfg["sigma_range"] = (lo, hi)
    cfg["sigma"] = 0.5 * (lo + hi)   # the scalar used at test time (conditioning only)
    cfg["p_clean"] = float(cfg["p_clean"])
    cfg["w_anom"] = float(cfg["w_anom"])
    cfg["mask_thr"] = float(cfg["mask_thr"])
    cfg["mask"] = str(cfg["mask"]).lower()
    cfg["kind"] = str(cfg["kind"]).lower()
    kinds = cfg["kinds"]
    if isinstance(kinds, str):
        kinds = kinds.replace(",", " ").split()
    cfg["kinds"] = [str(k).lower() for k in kinds]
    beta = cfg["beta"]
    if not isinstance(beta, (list, tuple)):
        beta = [beta, beta]
    cfg["beta"] = (float(beta[0]), float(beta[1]))
    cfg["dtd_bank"] = int(cfg["dtd_bank"])
    if cfg["kind"] not in CORRUPTION_KINDS + ("mix",):
        raise ValueError(f"noise.kind must be one of {CORRUPTION_KINDS + ('mix',)} "
                         f"(got '{cfg['kind']}')")
    if not cfg["kinds"] or any(k not in CORRUPTION_KINDS for k in cfg["kinds"]):
        raise ValueError(f"noise.kinds must be a non-empty subset of {CORRUPTION_KINDS} "
                         f"(got {cfg['kinds']})")
    if len(cfg["beta"]) != 2 or not 0.0 <= cfg["beta"][0] <= cfg["beta"][1] <= 1.0:
        raise ValueError(f"noise.beta must be [lo, hi] with 0 <= lo <= hi <= 1 (got {beta})")
    if not cfg["sigma_range"][0] > 0:
        raise ValueError(f"simplenet noise sigma must be > 0 (got {cfg['sigma_range']})")
    if cfg["mask"] not in ("perlin", "all"):
        raise ValueError(f"simplenet mask must be 'perlin' or 'all' (got '{cfg['mask']}')")
    if not 0.0 <= cfg["p_clean"] < 1.0:
        raise ValueError(f"simplenet p_clean must be in [0, 1) (got {cfg['p_clean']})")
    if cfg["w_anom"] < 1.0:
        raise ValueError(f"simplenet w_anom must be >= 1 (got {cfg['w_anom']})")
    return cfg


def sample_token_mask(
    z0: torch.Tensor,
    noise_cfg: Optional[Dict[str, Any]],
    size: int,
    generator: Optional[torch.Generator] = None,
    return_pixel: bool = False,
):
    """Per-token corruption weights ``m`` of shape ``(B, 1, h, w)`` in ``[0, 1]``.

    With ``return_pixel`` also returns the pixel mask ``M`` (``(B, 1, size,
    size)``, binary) that ``m`` was pooled from, for pixel-space corruptions.

    ``perlin``: a DRAEM-style Perlin blob drawn at ``size`` px (the image
    resolution) and average-pooled to the latent grid, so ``m`` is the fraction
    of each token's patch inside the blob.  ``all``: every token (SimpleNet).
    A ``p_clean`` fraction of samples gets ``m = 0``.  The numpy RNG is seeded
    from the torch RNG, so ``seed_all`` makes masks reproducible.
    """
    sn = simplenet_cfg(noise_cfg)
    b, _, h, w = z0.shape
    if generator is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    else:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator,
                                 device=generator.device).item())
    rng = np.random.default_rng(seed)
    if sn["mask"] == "all":
        m = torch.ones((b, 1, h, w))
        px = torch.ones((b, 1, int(size), int(size))) if return_pixel else None
    else:
        from latent_draem import perlin_mask  # same Perlin blobs as latent_draem / DRAEM
        pm = np.stack([perlin_mask(int(size), rng, thr=sn["mask_thr"]) for _ in range(b)])
        px = torch.from_numpy(pm.astype(np.float32)).unsqueeze(1)
        m = F.adaptive_avg_pool2d(px, (h, w))
    clean = torch.from_numpy(rng.random(b) < sn["p_clean"])
    m[clean] = 0.0
    m = m.to(device=z0.device, dtype=z0.dtype)
    if not return_pixel:
        return m
    px[clean] = 0.0
    return m, px.to(device=z0.device)


def sample_sigma(
    z0: torch.Tensor,
    noise_cfg: Optional[Dict[str, Any]],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-sample noise std, shape ``(B,)``: constant, or ``U(lo, hi)`` for a range."""
    lo, hi = simplenet_cfg(noise_cfg)["sigma_range"]
    b = z0.shape[0]
    if hi == lo:
        return torch.full((b,), lo, device=z0.device, dtype=z0.dtype)
    u = torch.rand(b, device=z0.device, dtype=z0.dtype) if generator is None else torch.rand(
        b, device=z0.device, dtype=z0.dtype, generator=generator)
    return lo + (hi - lo) * u


def simplenet_corrupt(
    z0: torch.Tensor,
    eps: torch.Tensor,
    sigma,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``z_in = z_0 + m * sigma * eps`` (untouched tokens stay exactly clean).

    ``sigma`` is a float or a per-sample tensor of shape ``(B,)``."""
    if torch.is_tensor(sigma):
        sigma = sigma.view(-1, *([1] * (z0.dim() - 1)))
    else:
        sigma = float(sigma)
    return z0 + mask * sigma * eps


# ---------------------------------------------------------------------------
# structured corruptions (swap / dtd / mix)

_TEXTURES: Optional[torch.Tensor] = None   # (N, 3, S, S) fp16 CPU, set by train


def needs_textures(noise_cfg: Optional[Dict[str, Any]]) -> bool:
    """Whether this noise block pastes DTD textures (needs a bank and the encoder)."""
    sn = simplenet_cfg(noise_cfg)
    return sn["kind"] == "dtd" or (sn["kind"] == "mix" and "dtd" in sn["kinds"])


def set_texture_bank(bank: Optional[torch.Tensor]) -> None:
    global _TEXTURES
    _TEXTURES = bank


def build_texture_bank(dtd_root, n: int, size: int, seed: int = 0) -> torch.Tensor:
    """``n`` random DTD crops at ``size`` px (latent_draem's crop + colour jitter).

    Downloads DTD (~600 MB, torchvision) into ``dtd_root`` if it holds no images."""
    import random
    from pathlib import Path

    from latent_draem import get_dtd_images, load_dtd_crop
    paths = get_dtd_images(Path(dtd_root))
    if not paths:
        raise FileNotFoundError(f"no DTD images under {dtd_root}")
    rng = random.Random(seed)
    pick = [rng.choice(paths) for _ in range(int(n))]
    return torch.stack([load_dtd_crop(p, int(size), rng) for p in pick]).half()


def _rand(n: int, ref: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.rand(n, device=ref.device)
    return torch.rand(n, device=ref.device, generator=generator)


def _randint(lo: int, hi: int, n: int, ref: torch.Tensor,
             generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randint(lo, hi, (n,), device=ref.device)
    return torch.randint(lo, hi, (n,), device=ref.device, generator=generator)


def latent_swap(
    z0: torch.Tensor,
    mask: torch.Tensor,
    beta: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """``z0 + beta*m*(src - z0)``: ``src`` is a batch member's latent, rolled by a
    random spatial shift and flipped half the time -- plausible features in the
    wrong place (latent_draem's ``latent_self``).  ``src`` is detached."""
    b, _, h, w = z0.shape
    j = _randint(0, b, b, z0, generator)
    lo_h, lo_w = (2 if h > 4 else 1), (2 if w > 4 else 1)
    sy = _randint(lo_h, max(h - lo_h + 1, lo_h + 1), b, z0, generator)
    sx = _randint(lo_w, max(w - lo_w + 1, lo_w + 1), b, z0, generator)
    flip = _rand(b, z0, generator) < 0.5
    src = []
    for k in range(b):
        s = torch.roll(z0[int(j[k])].detach(), shifts=(int(sy[k]), int(sx[k])), dims=(1, 2))
        src.append(s.flip(-1) if bool(flip[k]) else s)
    src = torch.stack(src)
    return z0 + beta.view(-1, 1, 1, 1).to(z0.dtype) * mask * (src - z0)


def texture_paste(
    images: torch.Tensor,
    pixel_mask: torch.Tensor,
    beta: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """DRAEM: ``x + beta*M*(texture - x)`` with a random bank texture (random flips)."""
    if _TEXTURES is None:
        raise RuntimeError("noise.kind needs DTD textures but no bank is loaded "
                           "(core.set_texture_bank; train() builds it from DTD_ROOT)")
    n, _, hh, ww = images.shape
    idx = _randint(0, _TEXTURES.shape[0], n, images, generator).cpu()
    tex = _TEXTURES[idx].to(device=images.device, dtype=images.dtype)
    if tex.shape[-2:] != (hh, ww):
        tex = F.interpolate(tex, size=(hh, ww), mode="bilinear", align_corners=False)
    fh = (_rand(n, images, generator) < 0.5).view(-1, 1, 1, 1)
    fv = (_rand(n, images, generator) < 0.5).view(-1, 1, 1, 1)
    tex = torch.where(fh, tex.flip(-1), tex)
    tex = torch.where(fv, tex.flip(-2), tex)
    pm = pixel_mask.to(images.dtype)
    if pm.shape[-2:] != (hh, ww):
        pm = F.interpolate(pm, size=(hh, ww), mode="nearest")
    return images + beta.view(-1, 1, 1, 1).to(images.dtype) * pm * (tex - images)


def structured_corrupt(
    z0: torch.Tensor,
    images: torch.Tensor,
    rae: Optional[torch.nn.Module],
    adaptor: Optional[torch.nn.Module],
    noise_cfg: Optional[Dict[str, Any]],
    size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Synthetic anomaly for the SimpleNet path, by ``noise.kind``.

    ``z0`` is the (adapted) clean latent.  Returns ``(z_in, m, t, kinds)``:
    ``m`` the token mask (loss weights and the DRAEM head's target), ``t`` the
    raw conditioning sigma per sample (its own sigma for Gaussian samples, the
    test-time midpoint for structured ones), ``kinds`` the kind per sample.
    Samples with an empty mask (``p_clean``) stay exactly ``z0``."""
    sn = simplenet_cfg(noise_cfg)
    b = z0.shape[0]
    mask, pmask = sample_token_mask(z0, noise_cfg, size, generator, return_pixel=True)
    if sn["kind"] == "mix":
        pool = list(sn["kinds"])
        kinds = [pool[int(i)] for i in _randint(0, len(pool), b, z0, generator)]
    else:
        kinds = [sn["kind"]] * b
    lo, hi = sn["beta"]
    beta = lo + (hi - lo) * _rand(b, z0, generator)
    t = torch.full((b,), sn["sigma"], device=z0.device, dtype=z0.dtype)
    zin = z0
    sel = lambda k: torch.tensor([kk == k for kk in kinds], device=z0.device)  # noqa: E731
    if "gaussian" in kinds:
        eps = torch.randn_like(z0) if generator is None else torch.randn(
            z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)
        sig = sample_sigma(z0, noise_cfg, generator)
        g = sel("gaussian")
        t = torch.where(g, sig, t)
        zin = torch.where(g.view(-1, 1, 1, 1), simplenet_corrupt(z0, eps, sig, mask), zin)
    if "swap" in kinds:
        s = sel("swap").view(-1, 1, 1, 1)
        zin = torch.where(s, latent_swap(z0, mask, beta, generator), zin)
    if "dtd" in kinds:
        if rae is None:
            raise RuntimeError("noise.kind 'dtd' needs the RAE encoder")
        ii = sel("dtd").nonzero().squeeze(1)
        xa = texture_paste(images[ii], pmask[ii], beta[ii], generator)
        with torch.no_grad():
            za = rae.encode(xa).to(z0.dtype)
        if adaptor is not None:
            za = adaptor(za)
        full = zin.clone()
        full[ii] = za
        zin = full
    clean = (mask.flatten(1).sum(1) == 0).view(-1, 1, 1, 1)
    zin = torch.where(clean, z0, zin)
    return zin, mask, t, kinds


def latent_restore_loss(
    zhat: torch.Tensor,
    z0: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    w_anom: float = 1.0,
) -> torch.Tensor:
    """Token MSE to the clean latent, corrupted tokens up-weighted by ``w_anom``.

    ``L = sum_p w_p * mean_c (z^_p - z_p)^2 / sum_p w_p``, ``w = 1 + (w_anom-1)*m``
    (``latent_draem.restore_loss`` weighting, MSE part only).
    """
    err = (zhat - z0).pow(2).mean(dim=1, keepdim=True)
    if mask is None:
        return err.mean()
    wgt = 1.0 + (float(w_anom) - 1.0) * mask
    return (err * wgt).sum() / wgt.sum()


#: latent-space anomaly scores (decoder-free scoring, experiment J)
LATENT_SCORES = ("latent_cos", "latent_l2")


def latent_anomaly_map(
    z: torch.Tensor,
    zhat: torch.Tensor,
    score: str,
    out_hw: Tuple[int, int],
) -> torch.Tensor:
    """Token-wise distance between the test latent and the DiT's restoration.

    ``latent_cos``: ``1 - cos(z_p, z^_p)`` per token (scale-invariant, the
    ``latent_draem`` residual score); ``latent_l2``: ``mean_c (z^_p - z_p)^2``.
    The ``(B, h, w)`` token map is bilinearly upsampled to ``out_hw`` so the
    rest of the evaluation (blur, AUROC, AUPRO) is unchanged.
    """
    if score == "latent_cos":
        d = 1.0 - F.cosine_similarity(zhat.float(), z.float(), dim=1, eps=1e-6)
    elif score == "latent_l2":
        d = (zhat.float() - z.float()).pow(2).mean(dim=1)
    else:
        raise ValueError(f"unknown latent score '{score}'; expected one of {LATENT_SCORES}")
    return F.interpolate(d.unsqueeze(1), size=tuple(out_hw), mode="bilinear",
                         align_corners=False)[:, 0]


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
    adaptor: Optional[torch.nn.Module] = None,
    lambda_latent: float = 0.0,
    disc: Optional[torch.nn.Module] = None,
    disc_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    """Build the training objective for one batch.

    ``loss_type``: ``flow_matching`` | ``image_l1`` | ``hybrid`` | ``latent``
    (decoder-free: ``latent_restore_loss(z_hat, z_0)``, experiment J).

    ``adaptor`` (optional): a feature adaptor ``A`` applied to the *clean*
    latents right after the encoder, i.e. ``z0 <- A(z0)``.  The adapted space is
    then the space the DiT models and the decoder decodes (both here and at
    inference), so no inverse of ``A`` is needed.  Identity-initialized
    adaptors therefore reproduce the baseline loss exactly at step 0.

    For ``flow_matching`` the untouched official ``training_losses`` is used
    (it draws its own time/noise).  For ``image_l1`` and ``hybrid`` a single
    shared ``(t, eps)`` draw and a single DiT forward feed both terms, so the
    hybrid's flow term is algebraically identical to the official loss.

    ``corruption='simplenet'`` (default for B-J) replaces the flow draw by
    ``z_in = z_0 + m*sigma*eps`` on a Perlin token mask (``sample_token_mask``)
    and is only valid with ``loss_type='image_l1'``.  ``lambda_latent > 0``
    adds ``lambda_latent * latent_restore_loss(z_hat, z_0)`` (corrupted tokens
    up-weighted by ``noise.w_anom`` under the simplenet corruption).

    ``disc`` (optional, ``iad.disc``): DRAEM's discriminative head on the token
    grid, trained with a focal loss against the SimpleNet mask.  Its loss is
    added to the returned tensor (for backward) but reported only as
    ``terms['seg']`` -- ``terms['total']`` stays the restorer objective, so the
    head cannot change checkpoint selection; with ``detach`` it cannot change
    the DiT's gradients either.

    Returns ``(total_loss, terms_dict, zt, zhat)``: ``terms_dict`` holds the
    detached scalar terms (``total``/``img``/``fm`` as applicable), ``zt`` is
    the corrupted latent of the shared draw, and ``zhat`` the clean-latent
    estimate (``None`` for the pure flow-matching branch).
    """
    terms: Dict[str, float] = {}
    z_enc = z0  # the encoder's latent, before any adaptor
    anchored = bool(getattr(adaptor, "is_anchored", False))
    if adaptor is not None:
        z0 = adaptor(z0)
    if corruption == "simplenet" and loss_type not in ("image_l1", "latent"):
        raise ValueError("corruption='simplenet' needs loss='image_l1' or 'latent': flow "
                         "matching and the hybrid need the linear flow path")
    if loss_type == "latent" and adaptor is not None and not anchored and any(
            p.requires_grad for p in adaptor.parameters()):
        raise ValueError("loss='latent' with a plain trainable adaptor is degenerate: "
                         "nothing anchors the adapted space, so the adaptor can shrink it "
                         "to win the loss. Use an anchored adaptor (preserve_norm: true, "
                         "inverse_head: true -- experiment K), a frozen one, or an image loss")
    if disc is not None and corruption != "simplenet":
        raise ValueError("the discriminator needs synthetic-anomaly masks: it requires "
                         "corruption='simplenet' (the flow path has no mask)")

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

    mask: Optional[torch.Tensor] = None
    if corruption == "simplenet":
        # SimpleNet-style: small Gaussian on a Perlin subset of tokens, graded
        # against the clean latent; conditioning time sigma / (1 + sigma)
        sn = simplenet_cfg(noise_cfg)
        if sn["kind"] == "gaussian":
            mask = sample_token_mask(z0, noise_cfg, size, generator)
            eps = torch.randn_like(z0) if generator is None else torch.randn(
                z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)
            t = sample_sigma(z0, noise_cfg, generator)   # raw sigma per sample
            zt = simplenet_corrupt(z0, eps, t, mask)
        else:
            # structured synthetic anomalies (swap / dtd / mix), graded the same way
            zt, mask, t, _ = structured_corrupt(z0, images, rae, adaptor, noise_cfg,
                                                size, generator)
    else:
        # shared draw for image / hybrid
        t, eps = sample_time(noise_cfg, transport, z0, generator)
        zt = corrupt_latent(z0, eps, t, mode=corruption)
    tc = conditioning_time(t, corruption)
    v = dit(zt, tc, y=y)
    zhat = prediction_to_clean_latent(zt, v, t, corruption=corruption)

    loss_img = None
    if loss_type in ("image_l1", "hybrid"):
        z_dec = adaptor.invert(zhat) if anchored else zhat  # back to the decoder's space
        loss_img, _ = image_reconstruction_loss(rae, z_dec, images, size, clamp=clamp_recon)
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
    elif loss_type == "latent":
        # decoder-free (experiment J): restore the clean latent directly; the
        # decoder is never called
        w_anom = simplenet_cfg(noise_cfg)["w_anom"] if mask is not None else 1.0
        total = latent_restore_loss(zhat, z0.detach(), mask, w_anom)
        terms["lat"] = float(total.detach())
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    if float(lambda_latent) > 0 and loss_type != "latent":
        # target detached: an adaptor must not shrink the target to win the term
        w_anom = simplenet_cfg(noise_cfg)["w_anom"] if mask is not None else 1.0
        loss_lat = latent_restore_loss(zhat, z0.detach(), mask, w_anom)
        terms["lat"] = float(loss_lat.detach())
        total = total + float(lambda_latent) * loss_lat

    if anchored and adaptor.lambda_inv > 0 and any(
            p.requires_grad for p in adaptor.parameters()):
        # invertibility anchor (experiment K): the adapted space must still hold z
        loss_inv = adaptor.anchor_loss(z_enc.detach(), z0)
        terms["inv"] = float(loss_inv.detach())
        total = total + adaptor.lambda_inv * loss_inv

    terms["total"] = float(total.detach())  # the restorer objective (no seg term)

    if disc is not None:
        # DRAEM's discriminative sub-network: predict the synthetic-anomaly mask
        # from (corrupted input, restoration)
        from iad.disc import disc_cfg as _disc_cfg, seg_loss as _seg_loss
        dc = _disc_cfg(disc_cfg)
        z_in, z_r = (zt.detach(), zhat.detach()) if dc["detach"] else (zt, zhat)
        loss_seg = _seg_loss(disc(z_in, z_r), mask, dc["gamma"])
        terms["seg"] = float(loss_seg.detach())
        total = total + dc["w_seg"] * loss_seg
    return total, terms, zt, zhat
