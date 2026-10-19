"""Tests A–I for the RAE-DiT IAD experiments (spec §7).

All tests are *weight-independent*: they run on CPU with tiny stand-in
modules (``TinyRAE`` / ``TinyDiT``) wherever the official 3.35 GB
checkpoints are not needed.  What they cover:

    A. shapes               — corruption / conversion / loss / inference shapes
    B. frozen encoder       — M.freeze_encoder + encoder_status
    C. frozen decoder       — freeze/unfreeze + optimizer param groups (0.1x lr)
    D. gradient flow        — image L1 reaches the DiT, never frozen decoder/encoder
    E. conversion algebra   — z_hat = z_t - t*v exactness (and eps endpoint)
    F. loss sanity          — finiteness, hybrid = img + lambda*fm,
                              shared-draw FM == official training_losses
    G. checkpoint roundtrip — save/load incl. EMA, decoder, optimizer, config
    H. inference modes      — single/multi/ode pass accounting, batch-size
                              invariant noise, image-score aggregations
    I. tiny overfit         — image L1 on a fixed batch actually decreases

Run:  pytest rae_dit_iad/tests/test_iad.py -v   (cluster env `rae` for torch)
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latent_draem import seed_all  # noqa: E402
from iad import core  # noqa: E402
from iad import models as M  # noqa: E402
from iad import infer as I  # noqa: E402
from iad import train as T  # noqa: E402


# ---------------------------------------------------------------------------
# tiny stand-ins (real architecture, miniature sizes)
# ---------------------------------------------------------------------------

class TinyRAE(nn.Module):
    """Identity-ish autoencoder: encode/decode are 1x1 convs with fixed

    identity weights, so ``decode(encode(x)) == x`` exactly.  Keeps the
    ``rae.encode`` / ``rae.decode`` / ``rae.encoder`` / ``rae.decoder``
    surface used by the production code.
    """

    def __init__(self) -> None:
        super().__init__()
        enc = nn.Conv2d(3, 3, 1)
        dec = nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            enc.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
            enc.bias.zero_()
            dec.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
            dec.bias.zero_()
        self.encoder = enc
        self.decoder = dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class TinyDiT(nn.Module):
    """Miniature DiT-shaped velocity net: ``(zt, t, y) -> v``."""

    def __init__(self, ch: int = 3, hidden: int = 16) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(ch, hidden, 3, padding=1)
        self.t_emb = nn.Linear(1, hidden)
        self.conv_out = nn.Conv2d(hidden, ch, 3, padding=1)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        h = self.conv_in(zt) + self.t_emb(t.view(-1, 1)).view(-1, -1, 1, 1)
        return self.conv_out(F.silu(h))


def _tiny_transport(shift: float = 1.0):
    """Official transport object (code-only; no checkpoint needed)."""
    pytest.importorskip("stage2.transport")
    from stage2.transport import create_transport
    return create_transport(path_type="Linear", prediction="velocity",
                            time_dist_type="logit-normal_0_1",
                            time_dist_shift=shift)


def _batch(b: int = 2, size: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((b, 3, size, size), generator=g)


NOISE = {"time_dist": "logit-normal_0_1", "warp": False}


# ---------------------------------------------------------------------------
# A. shapes
# ---------------------------------------------------------------------------

def test_a_shapes():
    z0 = _batch(2)
    eps = torch.randn_like(z0)
    t = torch.full((2,), 0.5)

    zt = core.corrupt_latent(z0, eps, t, mode="flow")
    assert zt.shape == z0.shape
    assert torch.allclose(zt, 0.5 * z0 + 0.5 * eps, atol=1e-6)

    zhat = core.prediction_to_clean_latent(zt, eps - z0, t)
    assert zhat.shape == z0.shape

    rae, dit, transport = TinyRAE(), TinyDiT(), _tiny_transport()
    y = M.null_labels(2, 1000, torch.device("cpu"))
    total, terms, zt_out, zhat_out = core.compute_losses(
        rae, dit, transport, z0, z0, loss_type="image_l1", noise_cfg=NOISE,
        lambda_fm=0.0, y=y, size=16)
    assert total.ndim == 0 and math.isfinite(float(total))
    assert zt_out.shape == zhat_out.shape == z0.shape
    assert set(terms) == {"total", "img"}

    # inference shapes
    maps_all = []
    x_img = _batch(3, 16, seed=1)
    for mode in ("single", "multi", "ode"):
        xhat, maps, meta = I.reconstruct(rae, dit, x_img, [0, 1, 2],
                                         mode=mode, t=0.5, repeats=3, ode_steps=4)
        assert xhat.shape == x_img.shape
        assert maps.shape == (3, 16, 16)
        assert meta["mode"] == mode
        maps_all.append(maps)
    # image score aggregations
    for agg in ("mean", "max", "top1"):
        s = I.image_score(maps_all[0].numpy(), agg)
        assert s.shape == (3,)
        assert np.isfinite(s).all()


# ---------------------------------------------------------------------------
# B. frozen encoder
# ---------------------------------------------------------------------------

def test_b_frozen_encoder():
    rae = TinyRAE()
    M.freeze_encoder(rae)
    assert M.encoder_status(rae)
    assert all(not p.requires_grad for p in rae.encoder.parameters())
    # unfreezing the decoder (Experiment C) must not touch the encoder
    M.unfreeze_decoder(rae)
    assert M.encoder_status(rae)


# ---------------------------------------------------------------------------
# C. frozen decoder + optimizer param groups
# ---------------------------------------------------------------------------

def test_c_frozen_decoder():
    rae, dit = TinyRAE(), TinyDiT()
    M.freeze_decoder(rae)
    assert all(not p.requires_grad for p in rae.decoder.parameters())
    assert all(p.requires_grad for p in dit.parameters())

    # frozen decoder -> single param group, official lr
    opt, base_lr, dec_lr, _ = M.build_optimizer(dit, rae, False, {"optimizer": {"lr": 2e-4}})
    assert dec_lr is None and len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4)

    # finetuned decoder -> two groups, decoder lr = 0.1 * base
    M.unfreeze_decoder(rae)
    opt, base_lr, dec_lr, _ = M.build_optimizer(
        dit, rae, True, {"optimizer": {"lr": 2e-4}, "decoder_lr_ratio": 0.1})
    assert len(opt.param_groups) == 2
    assert dec_lr == pytest.approx(2e-5)
    assert opt.param_groups[1]["lr"] == pytest.approx(2e-5)
    # group contents: group 0 = DiT params, group 1 = decoder params
    dit_ids = {id(p) for p in dit.parameters()}
    dec_ids = {id(p) for p in rae.decoder.parameters()}
    g0 = {id(p) for p in opt.param_groups[0]["params"]}
    g1 = {id(p) for p in opt.param_groups[1]["params"]}
    assert g0 == dit_ids and g1 == dec_ids


# ---------------------------------------------------------------------------
# D. gradient flow
# ---------------------------------------------------------------------------

def test_d_gradient_flow_image_l1():
    seed_all(0)
    rae, dit, transport = TinyRAE(), TinyDiT(), _tiny_transport()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 16, seed=3)
    z0 = rae.encode(x).detach()
    y = M.null_labels(2, 1000, torch.device("cpu"))

    total, _, _, _ = core.compute_losses(
        rae, dit, transport, x, z0, loss_type="image_l1", noise_cfg=NOISE,
        lambda_fm=0.0, y=y, size=16)
    total.backward()

    # DiT receives gradient through the frozen decoder
    dit_grads = [p.grad for p in dit.parameters() if p.grad is not None]
    assert dit_grads and all(torch.isfinite(g).all() for g in dit_grads)
    assert any(float(g.abs().sum()) > 0 for g in dit_grads)
    # frozen decoder/encoder: no parameter grads, but ops were differentiated
    # through (requires_grad of inputs unchanged, weights untouched)
    assert all(p.grad is None for p in rae.decoder.parameters())
    assert all(p.grad is None for p in rae.encoder.parameters())
    assert all(not p.requires_grad for p in rae.encoder.parameters())

    # Experiment C: finetuned decoder gets its own gradients
    M.unfreeze_decoder(rae)
    dit.zero_grad(set_to_none=True)
    total, _, _, _ = core.compute_losses(
        rae, dit, transport, x, z0, loss_type="image_l1", noise_cfg=NOISE,
        lambda_fm=0.0, y=y, size=16)
    total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in rae.decoder.parameters())
    assert all(p.grad is None for p in rae.encoder.parameters())


# ---------------------------------------------------------------------------
# E. prediction conversion algebra
# ---------------------------------------------------------------------------

def test_e_prediction_conversion_algebra():
    seed_all(1)
    z0 = _batch(4, 8, seed=5)
    eps = torch.randn_like(z0)
    for t_val in (0.0, 0.25, 0.5, 1.0):
        t = torch.full((4,), t_val)
        zt = core.corrupt_latent(z0, eps, t, mode="flow")
        v_star = eps - z0                       # official VELOCITY target
        zhat = core.prediction_to_clean_latent(zt, v_star, t)
        # exact data endpoint: z_t - t*v == z0 for the true velocity
        assert torch.allclose(zhat, z0, atol=1e-5), f"t={t_val}"
        # round-trip: z_hat + t*v == z_t
        assert torch.allclose(zhat + t.view(-1, 1, 1, 1) * v_star, zt, atol=1e-6)
        # noise endpoint exactness at the truth
        eps_hat = core.prediction_to_noise(zt, v_star, t)
        assert torch.allclose(eps_hat, eps, atol=1e-5), f"t={t_val}"

    # additive corruption: conditioning time = sigma/(1+sigma)
    t_add = torch.tensor([0.3])
    assert float(core.conditioning_time(t_add, "additive")) == pytest.approx(0.3 / 1.3)
    assert float(core.conditioning_time(t_add, "flow")) == pytest.approx(0.3)

    # official time warp: fixed points 0 and 1, monotone, matches the formula
    s = core.time_dist_shift_from_misc({"latent_size": [768, 16, 16]})
    assert s == pytest.approx(math.sqrt(48.0))
    for u in (0.0, 0.1, 0.5, 0.9, 1.0):
        w = float(core.warp_time(torch.tensor([u]), s))
        assert 0.0 <= w <= 1.0
        assert w == pytest.approx(s * u / (1 + (s - 1) * u))


# ---------------------------------------------------------------------------
# F. loss sanity
# ---------------------------------------------------------------------------

def test_f_loss_sanity():
    seed_all(2)
    rae, dit, transport = TinyRAE(), TinyDiT(), _tiny_transport()
    x = _batch(3, 16, seed=7)
    z0 = rae.encode(x).detach()
    y = M.null_labels(3, 1000, torch.device("cpu"))

    # every loss type is finite and non-negative (FM part can be 0 at init)
    for loss_type, lam in (("image_l1", 0.0), ("flow_matching", 0.0), ("hybrid", 0.25)):
        total, terms, _, _ = core.compute_losses(
            rae, dit, transport, x, z0, loss_type=loss_type, noise_cfg=NOISE,
            lambda_fm=lam, y=y, size=16)
        assert math.isfinite(float(total)), loss_type
        assert float(total) >= 0.0, loss_type
        assert set(terms) <= {"total", "img", "fm"}
        assert "total" in terms
        if loss_type == "hybrid":
            assert "img" in terms and "fm" in terms
            # hybrid = img + lambda * fm on the shared draw (within float error)
            expect = terms["img"] + lam * terms["fm"]
            assert terms["total"] == pytest.approx(expect, rel=1e-4, abs=1e-6)

    # shared-draw FM term == manual official formula mean_flat((v - (eps-z0))^2)
    t = torch.tensor([0.3, 0.6, 0.9])
    eps = torch.randn_like(z0)
    loss_shared, zt, ut, v = core.flow_matching_loss(transport, dit, z0, y, t, eps)
    v2 = dit(zt, t, y=y)
    manual = core.mean_flat((v2 - (eps - z0)) ** 2).mean()
    assert torch.allclose(loss_shared, manual, atol=1e-6)
    assert torch.allclose(ut, eps - z0, atol=1e-6)

    # patched official training_losses (fixed draw) == our shared-draw loss:
    # proves the untouched official path computes the same objective
    orig_sample = transport.sample
    transport.sample = lambda x1, *a, **k: (t, eps, x1)
    try:
        official = core.official_flow_matching_loss(transport, dit, z0, y)
    finally:
        transport.sample = orig_sample
    assert float(official) == pytest.approx(float(loss_shared), rel=1e-5, abs=1e-6)

    # image L1 is exactly mean |xhat - x| on a fresh forward
    zhat = core.prediction_to_clean_latent(zt, v, t)
    l1, recon = core.image_reconstruction_loss(rae, zhat, x, 16)
    assert torch.allclose(l1, F.l1_loss(recon, x), atol=1e-6)
    assert math.isfinite(float(l1))

    # unknown loss type / corruption rejected
    with pytest.raises(ValueError):
        core.compute_losses(rae, dit, transport, x, z0, loss_type="nope",
                            noise_cfg=NOISE, lambda_fm=0.0, y=y)
    with pytest.raises(ValueError):
        core.corrupt_latent(z0, eps, t, mode="bogus")


# ---------------------------------------------------------------------------
# G. checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_g_checkpoint_roundtrip(tmp_path):
    seed_all(3)
    rae, dit = TinyRAE(), TinyDiT()
    M.unfreeze_decoder(rae)
    ema = copy.deepcopy(dit)
    cfg = {"experiment": {"id": "G", "loss": "image_l1", "decoder_finetune": True,
                          "lambda_fm": 0.1, "corruption": "flow"},
           "training": {"use_ema": True}, "rae_config": "x.yaml", "size": 16}
    meta = {"categories": ["carpet"], "loss": "image_l1"}
    opt, base_lr, dec_lr, _ = M.build_optimizer(
        dit, rae, True, {"optimizer": {"lr": 2e-4}, "decoder_lr_ratio": 0.1})

    # perturb weights so the roundtrip is non-trivial
    with torch.no_grad():
        for p in dit.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    path = tmp_path / "ck.pt"
    T.save_checkpoint(path, step=123, dit=dit, ema=ema, rae=rae,
                      optimizer=opt, scheduler=None, cfg=cfg, meta=meta)

    state = T.load_checkpoint(path)
    assert state["step"] == 123
    assert state["config"] == cfg
    assert state["decoder"] is not None  # decoder_finetune=true saves it

    # fresh copies -> apply -> identical weights
    rae2, dit2 = TinyRAE(), TinyDiT()
    ema2, opt2 = copy.deepcopy(dit2), torch.optim.AdamW(
        [{"params": list(dit2.parameters())}, {"params": list(rae2.decoder.parameters())}],
        lr=2e-4)
    step = T.apply_checkpoint(state, dit2, ema2, rae2, opt2, None, restore_rng=False)
    assert step == 123
    for k, v in dit.state_dict().items():
        assert torch.equal(v, dit2.state_dict()[k]), k
    for k, v in ema.state_dict().items():
        assert torch.equal(v, ema2.state_dict()[k]), k
    for k, v in rae.decoder.state_dict().items():
        assert torch.equal(v, rae2.decoder.state_dict()[k]), k


# ---------------------------------------------------------------------------
# H. inference modes, batch-size invariance, scores
# ---------------------------------------------------------------------------

def test_h_inference_modes_and_scoring():
    seed_all(4)
    rae, dit = TinyRAE(), TinyDiT()
    x = _batch(4, 16, seed=9)

    # pass accounting
    _, _, meta = I.reconstruct(rae, dit, x, [0, 1, 2, 3], mode="single", t=0.5)
    assert meta["dit_passes"] == 1 and meta["decode_passes"] == 1
    _, _, meta = I.reconstruct(rae, dit, x, [0, 1, 2, 3], mode="multi", t=0.5, repeats=4)
    assert meta["dit_passes"] == 4 and meta["decode_passes"] == 4
    _, _, meta = I.reconstruct(rae, dit, x, [0, 1, 2, 3], mode="ode", t=0.5, ode_steps=7)
    assert meta["dit_passes"] == 7 and meta["decode_passes"] == 1
    with pytest.raises(AssertionError):
        I.reconstruct(rae, dit, x, [0], mode="nope")

    # ODE time grid: warp endpoints are fixed points only at 0/1, so a partial
    # integration must invert-warp its start — _inv_warp round-trips exactly
    # (regression test for the grid start point).
    s = core.time_dist_shift_from_misc({"latent_size": [768, 16, 16]})
    for u in (0.0, 0.13, 0.5, 1.0):
        w = float(core.warp_time(torch.tensor([u]), s))
        assert I._inv_warp(w, s) == pytest.approx(u, rel=1e-5)
    assert I._inv_warp(float(core.warp_time(torch.tensor([0.5]), s)), s) == pytest.approx(0.5)

    # batch-size invariance: image 2's reconstruction is identical whether it
    # is inferred alone or inside a batch (index-seeded noise)
    xa, _, _ = I.reconstruct(rae, dit, x[2:3], [2], mode="single", t=0.5, seed=11)
    xb, _, _ = I.reconstruct(rae, dit, x[:3], [0, 1, 2], mode="single", t=0.5, seed=11)
    assert torch.allclose(xa, xb[2:3], atol=1e-6)
    # repeatability across calls
    _, m1, _ = I.reconstruct(rae, dit, x, [0, 1, 2, 3], mode="multi", t=0.5, repeats=3, seed=11)
    _, m2, _ = I.reconstruct(rae, dit, x, [0, 1, 2, 3], mode="multi", t=0.5, repeats=3, seed=11)
    assert torch.equal(m1, m2)

    # image score aggregations on a known map
    maps = np.zeros((2, 100, 100), np.float32)
    maps[0, 0, 0] = 10.0
    maps[1] = 1.0
    assert I.image_score(maps, "max")[0] == pytest.approx(10.0)
    assert I.image_score(maps, "mean")[1] == pytest.approx(1.0)
    top1 = I.image_score(maps, "top1")[0]
    assert top1 > 1.0  # the outlier pixel is inside the top 1%
    with pytest.raises(ValueError):
        I.image_score(maps, "median")

    # blur: sigma=0 is identity; sigma>0 smooths
    blurred = I.blur_maps(maps, 0.0)
    assert np.allclose(blurred, maps)
    pytest.importorskip("scipy")
    smoothed = I.blur_maps(maps, 2.0)
    assert smoothed[0, 0, 0] < 10.0


# ---------------------------------------------------------------------------
# I. tiny overfit
# ---------------------------------------------------------------------------

def test_i_tiny_overfit():
    """One fixed batch, image L1: the loss must clearly decrease."""
    seed_all(5)
    rae, dit, transport = TinyRAE(), TinyDiT(), _tiny_transport()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 16, seed=13)
    z0 = rae.encode(x).detach()
    y = M.null_labels(2, 1000, torch.device("cpu"))
    noise = {"time_dist": "fixed", "t": 0.5, "warp": False}
    opt = torch.optim.Adam(dit.parameters(), lr=3e-3)

    losses = []
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        total, _, _, _ = core.compute_losses(
            rae, dit, transport, x, z0, loss_type="image_l1", noise_cfg=noise,
            lambda_fm=0.0, y=y, size=16)
        total.backward()
        opt.step()
        losses.append(float(total))

    head = float(np.mean(losses[5:15]))
    tail = float(np.mean(losses[-10:]))
    assert math.isfinite(tail)
    assert tail < 0.9 * head, f"loss did not decrease: head={head:.4f} tail={tail:.4f}"
