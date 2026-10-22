"""Feature adaptors placed between the frozen encoder and the generative model.

Motivation
----------
The RAE encoder (DINOv2-B) is frozen and was trained on ImageNet, so its
latents are not aligned with MVTec AD.  A small *feature adaptor* ``A`` is
inserted right after the encoder and defines the latent space that both the DiT
and the RAE decoder operate on::

    x --encoder--> z0 --A--> z~0 --corruption--> z_t --DiT--> z^~ --decoder--> x^

Training and inference use the *same* space, so no inverse of ``A`` is ever
needed: the decoder is either fine-tuned to accept adapted latents (experiment
G) or kept in-distribution by an identity-initialized residual adaptor
(experiment F).

The module family follows the feature-adaptation literature for AD:

* ``simplenet`` (default) — SimpleNet's Feature Adaptor: a bottleneck MLP
  ``Linear -> BatchNorm -> LeakyReLU -> Linear`` applied per patch token
  (Liu et al., "SimpleNet: A Simple Network for Image Anomaly Detection and
  Localization", CVPR 2023).
* ``mlp`` — LayerNorm + 2-layer MLP with GELU (feature-translator style, as in
  DFR/CFA-like adaptors).
* ``linear`` — a single linear map (learned feature rotation/scaling).
* ``conv1x1`` — 1x1 convolutional bottleneck on the ``(B, C, H, W)`` latent.
* ``affine`` — per-channel scale/bias (diagonal map).  With
  ``init: train_stats`` the scale/bias are *fit to the MVTec train-split latent
  statistics* (``z -> (z - mu) / sigma``), i.e. the training-free feature
  normalization used by PatchCore / PaDiM / FeaNorm-style pipelines; set
  ``trainable: false`` to keep them purely statistical.

Note on PatchCore: PatchCore itself learns no adaptor at all — it adapts by
greedy-coreset subsampling of nominal features plus nearest-neighbour scoring.
That mechanism is not applicable here (the decoder needs a continuous latent in
its input domain), so the closest analogue is offered as the ``affine`` /
``train_stats`` variant: statistics alignment fit on the target train split.

All learned variants default to ``residual: true`` with a zero-initialized
output head (``A(z) = z + h(z)``, ``h == 0`` at initialization), so training
starts *exactly* at the no-adaptor baseline (experiment F step 0 == B step 0)
and a frozen decoder stays in-distribution.  Statistics are always fit on the
MVTec **train split only** (no test leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

#: every accepted value of ``adaptor.type``
ADAPTOR_TYPES = ("none", "simplenet", "mlp", "linear", "conv1x1", "affine")


class FeatureAdaptor(nn.Module):
    """Token-wise feature adaptor ``A`` (``(B, C, H, W) -> (B, C, H, W)``)."""

    def __init__(
        self,
        dim: int = 768,
        type: str = "simplenet",
        rank: int = 4,
        residual: bool = True,
        zero_init: bool = True,
        norm: str = "batch",
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
        init: str = "identity",
        trainable: bool = True,
    ) -> None:
        super().__init__()
        type = str(type).lower()
        if type not in ADAPTOR_TYPES or type == "none":
            raise ValueError(f"unknown adaptor type '{type}'; expected one of {ADAPTOR_TYPES[1:]}")
        self.dim = int(dim)
        self.type = type
        self.residual = bool(residual)
        self.init = str(init).lower()
        if self.init not in ("identity", "train_stats"):
            raise ValueError(f"unknown adaptor init '{init}'; expected identity | train_stats")
        if self.init == "train_stats" and type != "affine":
            raise ValueError("adaptor init='train_stats' is only meaningful for type='affine' "
                             "(learned variants use init='identity')")
        self.hidden = max(8, int(dim) // max(int(rank), 1))
        if type != "affine" and not self.residual and zero_init:
            raise ValueError("zero_init=True requires residual=True "
                             "(a non-residual zero-initialized adaptor is identically 0)")

        self.net: Optional[nn.Module] = None
        self.scale: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        tail: Optional[nn.Module] = None

        if type == "simplenet":
            layers: list = [nn.Linear(self.dim, self.hidden)]
            if norm == "batch":
                layers.append(nn.BatchNorm1d(self.hidden))
            elif norm == "layer":
                layers.append(nn.LayerNorm(self.hidden))
            elif norm != "none":
                raise ValueError(f"unknown adaptor norm '{norm}'")
            layers += [nn.LeakyReLU(float(leaky_slope)), nn.Linear(self.hidden, self.dim)]
            self.net = nn.Sequential(*layers)
            tail = self.net[-1]
        elif type == "mlp":
            layers = []
            if norm == "batch":
                layers.append(nn.BatchNorm1d(self.dim))
            elif norm == "layer":
                layers.append(nn.LayerNorm(self.dim))
            elif norm != "none":
                raise ValueError(f"unknown adaptor norm '{norm}'")
            layers += [nn.Linear(self.dim, self.hidden), nn.GELU()]
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(self.hidden, self.dim))
            self.net = nn.Sequential(*layers)
            tail = self.net[-1]
        elif type == "linear":
            self.net = nn.Linear(self.dim, self.dim)
            tail = self.net
        elif type == "conv1x1":
            self.net = nn.Conv2d(self.dim, self.dim, 1)
            tail = self.net
        else:  # affine
            self.scale = nn.Parameter(torch.ones(self.dim))
            self.bias = nn.Parameter(torch.zeros(self.dim))

        if tail is not None and zero_init:
            nn.init.zeros_(tail.weight)
            if getattr(tail, "bias", None) is not None:
                nn.init.zeros_(tail.bias)

        # train-split statistics (filled by ``fit_train_stats``)
        self.register_buffer("train_mean", torch.zeros(self.dim), persistent=False)
        self.register_buffer("train_std", torch.ones(self.dim), persistent=False)
        self.register_buffer("stats_fitted", torch.zeros(1, dtype=torch.bool), persistent=False)

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    # -- properties --------------------------------------------------------

    @property
    def wants_stats(self) -> bool:
        """True when the adaptor has to see the target train-split statistics."""
        return self.init == "train_stats"

    @property
    def trainable(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def extra_repr(self) -> str:
        return (f"type={self.type}, dim={self.dim}, hidden={self.hidden}, "
                f"residual={self.residual}, init={self.init}, "
                f"trainable={self.trainable}, stats_fitted={bool(self.stats_fitted)}")

    # -- forward -----------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.type == "conv1x1":
            h = self.net(z)
        elif self.type == "affine":
            shape = (1, -1) + (1,) * (z.dim() - 2)
            h = z * self.scale.view(shape) + self.bias.view(shape)
            return h
        else:
            b, c, hgt, wid = z.shape
            tokens = z.permute(0, 2, 3, 1).reshape(b, hgt * wid, c)
            tokens = self.net(tokens)
            h = tokens.reshape(b, hgt, wid, c).permute(0, 3, 1, 2)
        return z + h if self.residual else h

    # -- train-split statistics (PatchCore/FeaNorm style) -------------------

    @torch.no_grad()
    def accumulate_stats(self, z: torch.Tensor) -> None:
        """Accumulate per-channel sums for ``z`` (any batch, fp32)."""
        zf = z.detach().float()
        flat = zf.reshape(zf.shape[0], zf.shape[1], -1).double()
        s = flat.sum(dim=(0, 2))
        sq = (flat * flat).sum(dim=(0, 2))
        n = int(flat.shape[0] * flat.shape[2])
        buf = getattr(self, "_stats_acc", None)
        if buf is None:
            self._stats_acc = [s, sq, torch.tensor(float(n), dtype=torch.float64)]
        else:
            buf[0] += s
            buf[1] += sq
            buf[2] += float(n)

    @torch.no_grad()
    def finalize_stats(self, eps: float = 1e-6) -> "FeatureAdaptor":
        buf = getattr(self, "_stats_acc", None)
        if buf is None:
            raise RuntimeError("finalize_stats() called before accumulate_stats()")
        s, sq, n = buf
        n = float(n)
        mean = s / n
        var = (sq / n - mean * mean).clamp_min_(0.0)
        self.train_mean.copy_(mean.float().to(self.train_mean.device))
        self.train_std.copy_(var.sqrt().clamp_min(eps).float().to(self.train_std.device))
        self.stats_fitted.fill_(True)
        del self._stats_acc
        return self

    @torch.no_grad()
    def init_from_stats(self) -> "FeatureAdaptor":
        """Apply the fitted statistics according to ``init``.

        ``identity``    -> ``scale = 1``, ``bias = 0``  (``A(z) = z``)
        ``train_stats`` -> ``scale = 1 / sigma``, ``bias = -mu / sigma``, i.e.
        the patch/feature normalization (whitening) of the train split.
        """
        if self.type != "affine":
            return self
        if not bool(self.stats_fitted):
            raise RuntimeError("init_from_stats() requires fit_train_stats() first")
        if self.init == "train_stats":
            inv = 1.0 / self.train_std
            self.scale.copy_(inv)
            self.bias.copy_(-self.train_mean * inv)
        else:  # identity
            self.scale.copy_(torch.ones_like(self.scale))
            self.bias.copy_(torch.zeros_like(self.bias))
        return self


def build_adaptor(
    cfg: Optional[Dict[str, Any]],
    dim: int = 768,
    device: Optional[torch.device] = None,
) -> Optional[FeatureAdaptor]:
    """Build the adaptor described by an experiment config block.

    ``adaptor: {type: none}`` (or a missing block) builds nothing.  Accepted
    keys: ``type``, ``rank``, ``residual``, ``zero_init``, ``norm``,
    ``leaky_slope``, ``dropout``, ``init``, ``trainable``, ``dim``.
    """
    cfg = dict(cfg or {})
    kind = str(cfg.pop("type", "none")).lower()
    if kind in ("none", "", "identity"):
        return None
    cfg.pop("dim", None)
    adaptor = FeatureAdaptor(dim=int(dim), type=kind, **cfg)
    if device is not None:
        adaptor = adaptor.to(device)
    return adaptor


@torch.no_grad()
def fit_train_stats(
    adaptor: FeatureAdaptor,
    latents: Dict[Path, torch.Tensor],
    paths: Iterable[Path],
    chunk: int = 64,
) -> Optional[FeatureAdaptor]:
    """Fit the adaptor's per-channel statistics on *train-split* latents only.

    Computed in float64 on CPU in chunks, so it does not touch GPU memory and
    is exact regardless of batch size.  Returns the adaptor (or None when no
    statistics are needed).
    """
    if adaptor is None or not adaptor.wants_stats:
        return adaptor
    paths = [Path(p) for p in paths]
    if not paths:
        raise RuntimeError("fit_train_stats: no train paths")
    for i in range(0, len(paths), int(chunk)):
        batch = torch.stack([latents[p] for p in paths[i:i + int(chunk)]])
        adaptor.accumulate_stats(batch)
    return adaptor.finalize_stats().init_from_stats()


def adaptor_meta(adaptor: Optional[FeatureAdaptor]) -> Optional[Dict[str, Any]]:
    """Provenance record written into checkpoints / ``meta``."""
    if adaptor is None:
        return None
    return {
        "type": adaptor.type,
        "dim": adaptor.dim,
        "hidden": adaptor.hidden,
        "residual": adaptor.residual,
        "init": adaptor.init,
        "trainable": adaptor.trainable,
        "stats_fitted": bool(adaptor.stats_fitted),
        "params": int(sum(p.numel() for p in adaptor.parameters())),
    }
