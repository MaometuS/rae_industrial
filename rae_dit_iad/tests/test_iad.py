"""Tests A–K for the RAE-DiT IAD experiments (spec §7).

Extra groups beyond the spec: J covers the memory switches (activation
recomputation, CPU EMA, encoder offload) and K the feature adaptor of
experiments F/G.

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
    J. memory switches      — checkpointed grads == plain grads, CPU EMA /
                              encoder offload are exact
    K. feature adaptor      — identity at init (F step 0 == B), grads reach the
                              adaptor, train-stats init, checkpoint roundtrip,
                              inference uses the same adapted latent space
    L. SimpleNet corruption — masked small-sigma noise leaves clean tokens
                              untouched, reproducible masks, oracle exactness,
                              loss = img + lambda_latent*lat, guards, and no
                              test-time noise at inference (default corruption)
    M. decoder-free arm     — loss='latent' never calls the decoder, equals the
                              token MSE, rejects a trainable adaptor; latent
                              anomaly maps (cos / l2) and their use in inference
    N. discriminator        — DRAEM's segmentation head: leaves the DiT's gradients
                              and the reported total unchanged (detach), learns
                              from the mask, needs SimpleNet masks, disc score
    O. anchored adaptor     — K starts exactly at J, keeps token norms, the
                              invertibility anchor penalizes information loss,
                              plain adaptors still refused, pixel decode via B
    P. noise range, EMA     — sigma ~ U(lo, hi) per sample (exact per-sample
                              noise, midpoint at test), validation never moves a
                              GPU-resident EMA to the CPU
    Q. structured anomalies — swap moves real tokens, dtd pastes a bank texture
                              in pixel space and re-encodes, mix assigns one kind
                              per sample, clean samples stay exact, J3-J5 run
    R. per-category disk    — eval-only checkpoints hold the EMA weights and no
                              optimizer; slimming keeps the weights
    S. EMA correctness      — the EMA copy mirrors BatchNorm running statistics
                              (buffers), so it computes the trained function in
                              eval mode; CPU-resident EMAs too

Run:  pytest rae_dit_iad/tests/test_iad.py -v   (cluster env `rae` for torch)
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latent_draem import seed_all  # noqa: E402
from iad import adaptor as A  # noqa: E402
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
        h = self.conv_in(zt) + self.t_emb(t.view(-1, 1)).view(t.shape[0], -1, 1, 1)
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
    # top 1% of 10,000 pixels = the 100 highest: the outlier (10) + 99 zeros -> 0.1,
    # i.e. 100x the plain mean (0.001): the outlier dominates the top-1% score
    assert top1 == pytest.approx(10.0 / 100)
    assert top1 > I.image_score(maps, "mean")[0] * 50
    assert I.image_score(maps, "top1")[1] == pytest.approx(1.0)
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


# ---------------------------------------------------------------------------
# J. memory-only switches (gradient checkpointing + encoder offload)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Official-style DiT block: tensor kwargs (``feat_rope``) are passed through."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, feat_rope: torch.Tensor = None) -> torch.Tensor:
        out = self.act(self.conv(h))
        return out if feat_rope is None else out + feat_rope


class BlockDiT(nn.Module):
    """DiT-shaped net with a ``blocks`` ModuleList, as the official DiT has."""

    def __init__(self, ch: int = 3, hidden: int = 8, n_blocks: int = 3) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(ch, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(n_blocks)])
        self.rope = torch.zeros(1)          # non-grad tensor, like feat_rope
        self.conv_out = nn.Conv2d(hidden, ch, 3, padding=1)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        h = self.conv_in(zt) + t.view(-1, 1, 1, 1)
        for blk in self.blocks:
            h = blk(h, feat_rope=self.rope)   # kwargs path through the wrapper
        return self.conv_out(h)


def test_j_memory_switches():
    """Checkpointing must not change gradients; encoder offload must be exact."""
    seed_all(7)
    rae = TinyRAE()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 16, seed=21)
    z0 = rae.encode(x).detach()

    # (a) same loss/grads with and without activation recomputation
    def run(checkpoint: bool):
        seed_all(11)
        dit = BlockDiT()
        if checkpoint:
            wrapped = M.enable_gradient_checkpointing(dit, rae)
            assert wrapped == len(dit.blocks)
        t = torch.full((2,), 0.5)
        v = dit(z0, t)
        loss = (v ** 2).mean()
        loss.backward()
        grads = [p.grad.detach().clone() for p in dit.parameters()]
        return float(loss), grads

    loss_plain, grads_plain = run(False)
    loss_ckpt, grads_ckpt = run(True)
    assert loss_plain == pytest.approx(loss_ckpt, rel=1e-6, abs=1e-8)
    for a, b in zip(grads_plain, grads_ckpt):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), "checkpointing changed gradients"

    # (b) no_grad / eval path still runs, and wrapping is idempotent
    dit = BlockDiT()
    assert M.enable_gradient_checkpointing(dit, rae) == len(dit.blocks)
    assert M.enable_gradient_checkpointing(dit, rae) == len(dit.blocks)
    with torch.no_grad():
        assert dit(z0, torch.full((2,), 0.5)).shape == z0.shape
        assert rae.decode(z0).shape == x.shape

    # (c) decoder fine-tuning still receives gradients through the wrapper
    dit = BlockDiT()
    M.enable_gradient_checkpointing(dit, rae)
    M.unfreeze_decoder(rae)
    recon = rae.decode(dit(z0, torch.full((2,), 0.5)))
    (recon ** 2).mean().backward()
    assert all(p.grad is not None for p in rae.decoder.parameters())

    # (d) offload is exact: same latents, encoder on CPU afterwards
    rae2 = TinyRAE()
    M.freeze_encoder(rae2)
    before = rae2.encode(x)
    M.offload_encoder(rae2)
    assert next(rae2.encoder.parameters()).device.type == "cpu"
    after = rae2.encode(x)
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# K. feature adaptor between the frozen encoder and the DiT/decoder (F/G)
# ---------------------------------------------------------------------------

def _img_l1_loss(rae, dit, transport, x, z0, adaptor=None, seed=3):
    """One image-L1 loss value under a fixed draw (and the DiT grads)."""
    seed_all(seed)
    y = M.null_labels(z0.shape[0], 1000, torch.device("cpu"))
    total, _, _, _ = core.compute_losses(
        rae, dit, transport, x, z0, loss_type="image_l1",
        noise_cfg={"time_dist": "fixed", "t": 0.5, "warp": False},
        lambda_fm=0.0, y=y, size=16, adaptor=adaptor)
    total.backward()
    grads = [p.grad.detach().clone() for p in dit.parameters() if p.grad is not None]
    return float(total), grads


def test_k_feature_adaptor_identity_and_loss_equivalence():
    """Identity-initialized adaptors must not change the baseline at all."""
    seed_all(17)
    rae, dit, transport = TinyRAE(), TinyDiT(), _tiny_transport()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 16, seed=31)
    z0 = rae.encode(x).detach()

    baseline, grads_base = _img_l1_loss(rae, dit, transport, x, z0)
    for kind in ("simplenet", "mlp", "linear", "conv1x1", "affine"):
        ad = A.build_adaptor({"type": kind, "rank": 2}, dim=z0.shape[1])
        assert ad is not None and ad.type == kind
        # (a) A(z) == z exactly: residual head / affine are zero-initialized
        assert torch.equal(ad(z0), z0), f"{kind} is not identity at init"
        # (b) identical loss and identical DiT grads vs. no-adaptor
        for p in dit.parameters():
            p.grad = None
        loss, grads = _img_l1_loss(rae, dit, transport, x, z0, adaptor=ad)
        assert loss == pytest.approx(baseline, rel=1e-12, abs=1e-12), kind
        for a, b in zip(grads_base, grads):
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-8), kind

    # (c) gradients reach the adaptor and one step moves it off identity
    ad = A.build_adaptor({"type": "simplenet", "rank": 2}, dim=z0.shape[1])
    opt = torch.optim.Adam(ad.parameters(), lr=1e-2)
    for p in dit.parameters():
        p.grad = None
    _img_l1_loss(rae, dit, transport, x, z0, adaptor=ad)
    grads = [p.grad for p in ad.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), \
        "no gradient reached the adaptor"
    opt.step()
    assert not torch.equal(ad(z0), z0), "adaptor did not move off identity"

    # (d) train_stats affine = PatchCore/FeaNorm-style standardization
    stats_ad = A.build_adaptor({"type": "affine", "init": "train_stats"}, dim=z0.shape[1])
    A.fit_train_stats(stats_ad, {Path("a"): z0[0], Path("b"): z0[1]}, [Path("a"), Path("b")])
    flat = z0.transpose(1, 0).reshape(z0.shape[1], -1).double()
    assert torch.allclose(stats_ad.train_mean.double(), flat.mean(1), rtol=1e-5, atol=1e-6)
    assert torch.allclose(stats_ad.train_std.double(), flat.std(1, unbiased=False),
                          rtol=1e-4, atol=1e-6)
    expected = (z0 - stats_ad.train_mean.view(1, -1, 1, 1)) / \
        stats_ad.train_std.view(1, -1, 1, 1)
    assert torch.allclose(stats_ad(z0), expected, rtol=1e-5, atol=1e-6)
    frozen_ad = A.build_adaptor({"type": "affine", "init": "train_stats", "trainable": False},
                                dim=z0.shape[1])
    assert not frozen_ad.trainable and frozen_ad.trainable is False

    # (e) "none" builds nothing; bad types fail loudly
    assert A.build_adaptor(None) is None
    assert A.build_adaptor({"type": "none"}) is None
    with pytest.raises(ValueError):
        A.build_adaptor({"type": "does-not-exist"})
    with pytest.raises(ValueError):
        A.build_adaptor({"type": "linear", "residual": False, "zero_init": True})


def test_k_feature_adaptor_checkpoint_and_inference(tmp_path):
    """Adaptor weights must survive the checkpoint roundtrip and be used at inference."""
    seed_all(23)
    rae, dit, transport = TinyRAE(), TinyDiT(), _tiny_transport()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 16, seed=41)
    z0 = rae.encode(x).detach()

    ad = A.build_adaptor({"type": "simplenet", "rank": 2}, dim=z0.shape[1])
    with torch.no_grad():                      # make it non-trivial
        ad.net[-1].weight.normal_(0, 0.05)
    ema_ad = copy.deepcopy(ad)
    ema = copy.deepcopy(dit)

    cfg = {"experiment": {"decoder_finetune": False}, "adaptor": {"type": "simplenet", "rank": 2}}
    meta = {"adaptor": A.adaptor_meta(ad)}
    opt = torch.optim.AdamW(list(dit.parameters()) + list(ad.parameters()), lr=1e-4)
    path = tmp_path / "tiny.pt"
    T.save_checkpoint(path, 7, dit, ema, rae, opt, None, cfg, meta,
                      adaptor=ad, ema_adaptor=ema_ad)
    state = T.load_checkpoint(path)
    assert state["adaptor"] is not None and state["ema_adaptor"] is not None
    assert state["meta"]["adaptor"]["type"] == "simplenet"
    assert state["meta"]["adaptor"]["params"] == sum(p.numel() for p in ad.parameters())

    rae2, dit2 = TinyRAE(), TinyDiT()
    ad2 = A.build_adaptor(cfg["adaptor"], dim=z0.shape[1])
    ad2_ema = A.build_adaptor(cfg["adaptor"], dim=z0.shape[1])
    T.apply_checkpoint(state, dit2, None, rae2, None, None, restore_rng=False,
                       adaptor=ad2, ema_adaptor=ad2_ema)
    for a, b in zip(ad.state_dict().values(), ad2.state_dict().values()):
        assert torch.equal(a, b)
    assert torch.equal(ad2(z0), ad2_ema(z0))

    # inference uses the adaptor: same convention as training => identical to
    # the no-adaptor path when the adaptor is identity, different otherwise
    ident = A.build_adaptor({"type": "simplenet", "rank": 2}, dim=z0.shape[1])
    idx = [0, 1]
    _, maps_none, meta_none = I.reconstruct(rae, dit, x, idx, mode="single", t=0.5, seed=5)
    _, maps_ident, meta_ident = I.reconstruct(rae, dit, x, idx, mode="single", t=0.5, seed=5,
                                              adaptor=ident)
    assert meta_none["adaptor"] is None and meta_ident["adaptor"] == "simplenet"
    assert torch.allclose(maps_none, maps_ident, rtol=0, atol=0)
    _, maps_ad, _ = I.reconstruct(rae, dit, x, idx, mode="single", t=0.5, seed=5, adaptor=ad)
    assert not torch.allclose(maps_none, maps_ad)


# ---------------------------------------------------------------------------
# L. SimpleNet-style corruption (project default)
# ---------------------------------------------------------------------------

def test_l_simplenet_corruption_and_conversion():
    """z_in = z0 + m*sigma*eps: exact on masked tokens, untouched elsewhere."""
    seed_all(5)
    z0 = torch.randn(4, 8, 16, 16)
    # (a) mask=all, no clean samples: the added noise is exactly sigma * eps
    cfg_all = {"sigma": 0.1, "mask": "all", "p_clean": 0.0}
    m = core.sample_token_mask(z0, cfg_all, size=16)
    assert m.shape == (4, 1, 16, 16) and torch.all(m == 1)
    eps = torch.randn_like(z0)
    zt = core.simplenet_corrupt(z0, eps, 0.1, m)
    assert torch.allclose(zt - z0, 0.1 * eps, atol=1e-6)  # float32 rounding of z0 + d - z0

    # (b) Perlin mask: soft token coverage in [0, 1]; untouched tokens stay exactly
    #     clean; p_clean samples are fully clean
    cfg = {"sigma": 0.1, "mask": "perlin", "p_clean": 0.5}
    zb = torch.randn(16, 8, 16, 16)
    seed_all(6)
    mp = core.sample_token_mask(zb, cfg, size=64)
    assert mp.shape == (16, 1, 16, 16)
    assert float(mp.min()) >= 0.0 and float(mp.max()) <= 1.0
    clean_rows = mp.flatten(1).sum(1) == 0
    assert clean_rows.any() and (~clean_rows).any()
    ztp = core.simplenet_corrupt(zb, torch.randn_like(zb), 0.1, mp)
    untouched = (mp == 0).expand_as(zb)
    assert torch.equal(ztp[untouched], zb[untouched])

    # (c) reproducible under seed_all (numpy RNG is seeded from torch)
    seed_all(6)
    assert torch.equal(core.sample_token_mask(zb, cfg, size=64), mp)

    # (d) conditioning time sigma/(1+sigma) and oracle exactness z_hat == z0
    t = torch.full((4,), 0.1)
    tc = core.conditioning_time(t, "simplenet")
    assert torch.allclose(tc, t / 1.1)
    v_star = (zt - z0) / tc.view(-1, 1, 1, 1)
    zhat = core.prediction_to_clean_latent(zt, v_star, t, corruption="simplenet")
    assert torch.allclose(zhat, z0, atol=1e-6)

    # (e) bad configs fail loudly
    for bad in ({"sigma": 0.0}, {"mask": "box"}, {"p_clean": 1.0}, {"w_anom": 0.5},
                {"sigma": [1.5, 0.5]}, {"sigma": [0.0, 1.0]}, {"sigma": [0.5, 1.0, 1.5]},
                {"kind": "cutpaste"}, {"kinds": ["gaussian", "box"]}, {"kinds": []},
                {"beta": [0.8, 0.2]}, {"beta": [0.5, 1.5]}):
        with pytest.raises(ValueError):
            core.simplenet_cfg(bad)


def test_l_simplenet_losses_and_guards():
    """B/D objectives: finite, img + lambda_latent*lat, gradients reach the DiT."""
    seed_all(7)
    rae, dit = TinyRAE(), TinyDiT()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 32, seed=3)
    z0 = rae.encode(x).detach()
    y = torch.full((2,), 1000, dtype=torch.long)
    noise = {"sigma": 0.1, "mask": "perlin", "p_clean": 0.0, "w_anom": 5.0}
    kw = dict(noise_cfg=noise, lambda_fm=0.0, y=y, corruption="simplenet", size=32)

    total, terms, zt, zhat = core.compute_losses(rae, dit, None, x, z0,
                                                 loss_type="image_l1", lambda_latent=0.5, **kw)
    assert math.isfinite(float(total)) and zhat.shape == z0.shape
    assert terms["total"] == pytest.approx(terms["img"] + 0.5 * terms["lat"], rel=1e-6)
    total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in dit.parameters())
    assert all(p.grad is None for p in rae.parameters()), "frozen RAE received gradients"

    # lambda_latent = 0 is the pure image-L1 objective (experiment H)
    _, terms0, _, _ = core.compute_losses(rae, dit, None, x, z0,
                                          loss_type="image_l1", lambda_latent=0.0, **kw)
    assert "lat" not in terms0 and terms0["total"] == pytest.approx(terms0["img"])

    # weighted latent loss up-weights corrupted tokens
    err = torch.zeros(1, 2, 2, 2)
    err[..., 0, 0] = 1.0  # error only on token (0, 0)
    msk = torch.zeros(1, 1, 2, 2)
    msk[..., 0, 0] = 1.0
    assert float(core.latent_restore_loss(err, torch.zeros_like(err), msk, 5.0)) == \
        pytest.approx(5.0 / 8.0)
    assert float(core.latent_restore_loss(err, torch.zeros_like(err))) == pytest.approx(0.25)

    # guards: flow matching and the hybrid need the linear flow path
    for lt in ("flow_matching", "hybrid"):
        with pytest.raises(ValueError):
            core.compute_losses(rae, dit, None, x, z0, loss_type=lt, **kw)


def test_l_simplenet_inference_adds_no_noise():
    """At test time the real defect is the only corruption: zt == E(x)."""

    class ZeroDiT(nn.Module):
        def forward(self, zt, t, y=None):
            return torch.zeros_like(zt)

    rae = TinyRAE()
    x = _batch(3, 16, seed=9)
    xhat, maps, meta = I.reconstruct(rae, ZeroDiT(), x, [0, 1, 2], mode="multi", t=0.1,
                                     repeats=4, corruption="simplenet", seed=0)
    assert meta["reps"] == 1 and meta["dit_passes"] == 1 and meta["corruption"] == "simplenet"
    # zero velocity => z_hat = z_in = E(x) => identity decode (no noise was added)
    assert torch.allclose(xhat, x.clamp(0, 1), atol=1e-6)
    assert float(maps.abs().max()) < 1e-5
    with pytest.raises(AssertionError):
        I.reconstruct(rae, ZeroDiT(), x, [0, 1, 2], mode="ode", t=0.1, corruption="simplenet")


# ---------------------------------------------------------------------------
# M. decoder-free arm (experiment J)
# ---------------------------------------------------------------------------

class _NoDecodeRAE(TinyRAE):
    def decode(self, z):  # noqa: D401
        raise AssertionError("the decoder must not be called by the decoder-free arm")


def test_m_latent_loss_never_decodes():
    """loss='latent': token MSE to the clean latent, decoder untouched."""
    seed_all(11)
    rae, dit = _NoDecodeRAE(), TinyDiT()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 32, seed=5)
    z0 = rae.encode(x).detach()
    y = torch.full((2,), 1000, dtype=torch.long)
    kw = dict(lambda_fm=0.0, y=y, corruption="simplenet", size=32)

    # (a) mask=all: uniform weights, so the loss is the plain token MSE; the
    #     image term is absent and lambda_latent is ignored (no double count)
    noise_all = {"sigma": 0.1, "mask": "all", "p_clean": 0.0, "w_anom": 5.0}
    total, terms, _, zhat = core.compute_losses(rae, dit, None, x, z0, loss_type="latent",
                                                noise_cfg=noise_all, lambda_latent=3.0, **kw)
    assert set(terms) == {"lat", "total"}
    assert terms["total"] == pytest.approx(terms["lat"])
    assert terms["lat"] == pytest.approx(float(((zhat - z0) ** 2).mean()), rel=1e-5)
    total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in dit.parameters())

    # (b) Perlin mask: still finite, still no decoder call
    noise = {"sigma": 0.1, "mask": "perlin", "p_clean": 0.0, "w_anom": 5.0}
    total, _, _, _ = core.compute_losses(rae, dit, None, x, z0, loss_type="latent",
                                         noise_cfg=noise, **kw)
    assert math.isfinite(float(total))

    # (c) a trainable adaptor is degenerate without a decoder anchor -> refused;
    #     a frozen (statistics-only) adaptor is allowed
    ad = A.build_adaptor({"type": "simplenet", "rank": 2}, dim=z0.shape[1])
    with pytest.raises(ValueError):
        core.compute_losses(rae, dit, None, x, z0, loss_type="latent", noise_cfg=noise,
                            adaptor=ad, **kw)
    frozen = A.build_adaptor({"type": "affine", "trainable": False}, dim=z0.shape[1])
    total, _, _, _ = core.compute_losses(rae, dit, None, x, z0, loss_type="latent",
                                         noise_cfg=noise, adaptor=frozen, **kw)
    assert math.isfinite(float(total))


def test_m_latent_loss_with_flow_corruption():
    """loss='latent' also works on the flow path (no decoder call there either)."""
    transport = _tiny_transport()
    seed_all(12)
    rae, dit = _NoDecodeRAE(), TinyDiT()
    x = _batch(2, 16, seed=6)
    z0 = rae.encode(x).detach()
    y = torch.full((2,), 1000, dtype=torch.long)
    total, terms, _, zhat = core.compute_losses(rae, dit, transport, x, z0, loss_type="latent",
                                                noise_cfg=NOISE, lambda_fm=0.0, y=y,
                                                corruption="flow", size=16)
    assert terms["lat"] == pytest.approx(float(((zhat - z0) ** 2).mean()), rel=1e-5)


def test_m_latent_anomaly_maps():
    """Token distances: zero at a perfect restoration, located at the changed token."""
    torch.manual_seed(13)
    z = torch.randn(2, 8, 4, 4)
    m = core.latent_anomaly_map(z, z.clone(), "latent_cos", (16, 16))
    assert m.shape == (2, 16, 16) and float(m.abs().max()) < 1e-5

    zh = z.clone()
    zh[:, :, 1, 2] = -z[:, :, 1, 2]  # flip one token
    d = core.latent_anomaly_map(z, zh, "latent_cos", (4, 4))  # same size: no resampling
    assert torch.allclose(d[:, 1, 2], torch.full((2,), 2.0), atol=1e-5)
    d[:, 1, 2] = 0
    assert float(d.abs().max()) < 1e-5

    # cos is scale-invariant, l2 is not
    assert float(core.latent_anomaly_map(z, 3 * z, "latent_cos", (4, 4)).abs().max()) < 1e-5
    l2 = core.latent_anomaly_map(z, 3 * z, "latent_l2", (4, 4))
    assert torch.allclose(l2, (4 * z ** 2).mean(1), rtol=1e-5)
    with pytest.raises(ValueError):
        core.latent_anomaly_map(z, z, "pixel", (4, 4))


def test_m_reconstruct_with_latent_score():
    """inference: latent score compares E(x) with the DiT output, before the decoder."""

    class ZeroDiT(nn.Module):
        def forward(self, zt, t, y=None):
            return torch.zeros_like(zt)

    class NegDiT(nn.Module):  # v = -z  =>  z^ = (1 + t_c) z
        def forward(self, zt, t, y=None):
            return -zt

    rae = TinyRAE()
    x = _batch(2, 16, seed=14)
    _, maps, meta = I.reconstruct(rae, ZeroDiT(), x, [0, 1], t=0.1, corruption="simplenet",
                                  score="latent_cos")
    assert meta["score"] == "latent_cos" and maps.shape == (2, 16, 16)
    assert float(maps.abs().max()) < 1e-5
    _, cos_maps, _ = I.reconstruct(rae, NegDiT(), x, [0, 1], t=0.1, corruption="simplenet",
                                   score="latent_cos")
    _, l2_maps, _ = I.reconstruct(rae, NegDiT(), x, [0, 1], t=0.1, corruption="simplenet",
                                  score="latent_l2")
    assert float(cos_maps.abs().max()) < 1e-5 and float(l2_maps.min()) > 0
    _, _, meta_px = I.reconstruct(rae, ZeroDiT(), x, [0, 1], t=0.1, corruption="simplenet")
    assert meta_px["score"] == "pixel"
    with pytest.raises(AssertionError):
        I.reconstruct(rae, ZeroDiT(), x, [0, 1], t=0.1, corruption="simplenet", score="bogus")


# ---------------------------------------------------------------------------
# N. discriminator (DRAEM's discriminative sub-network, token grid)
# ---------------------------------------------------------------------------

_DISC = {"enabled": True, "d": 16, "blocks": 1}


def test_n_discriminator_leaves_dit_training_unchanged():
    """detach: the head learns from the DiT but never changes the DiT's gradients."""
    from iad import disc as DS
    rae, dit = TinyRAE(), TinyDiT()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 32, seed=21)
    z0 = rae.encode(x).detach()
    y = torch.full((2,), 1000, dtype=torch.long)
    kw = dict(loss_type="image_l1", noise_cfg={"sigma": 0.1, "mask": "perlin", "p_clean": 0.0},
              lambda_fm=0.0, y=y, corruption="simplenet", size=32, lambda_latent=1.0)

    seed_all(3)
    tot0, terms0, _, _ = core.compute_losses(rae, dit, None, x, z0, **kw)
    tot0.backward()
    g0 = [p.grad.clone() for p in dit.parameters()]
    for p in dit.parameters():
        p.grad = None

    head = DS.build_discriminator(_DISC, dim=z0.shape[1], seed=0)
    seed_all(3)
    tot1, terms1, _, _ = core.compute_losses(rae, dit, None, x, z0, disc=head,
                                             disc_cfg=_DISC, **kw)
    tot1.backward()
    # reported restorer objective unchanged; seg reported separately, added for backward
    assert terms1["total"] == pytest.approx(terms0["total"], rel=1e-6)
    assert "seg" in terms1 and math.isfinite(terms1["seg"])
    assert float(tot1) == pytest.approx(terms1["total"] + terms1["seg"], rel=1e-5)
    # identical DiT gradients, and the head itself learns
    for a, p in zip(g0, dit.parameters()):
        assert torch.allclose(a, p.grad, rtol=1e-6, atol=1e-8)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in head.parameters())


def test_n_discriminator_config_rng_and_guards():
    from iad import disc as DS
    assert DS.build_discriminator(None, dim=3) is None
    assert DS.build_discriminator({"enabled": False}, dim=3) is None
    state = torch.get_rng_state()
    head = DS.build_discriminator(_DISC, dim=3, seed=5)
    assert head is not None and torch.equal(state, torch.get_rng_state()), \
        "building the head must not advance the global RNG"
    same = DS.build_discriminator(_DISC, dim=3, seed=5)
    for a, b in zip(head.parameters(), same.parameters()):
        assert torch.equal(a, b)
    for bad in ({"d": 12}, {"bogus": 1}, {"w_seg": 0.0}):
        with pytest.raises(ValueError):
            DS.disc_cfg(bad)
    # focal loss: confident and right << confident and wrong
    tgt = torch.ones(1, 1, 2, 2)
    assert float(DS.seg_loss(torch.full_like(tgt, 5.0), tgt)) < \
        float(DS.seg_loss(torch.full_like(tgt, -5.0), tgt))

    # the flow path has no synthetic-anomaly mask -> refused
    rae, dit = TinyRAE(), TinyDiT()
    x = _batch(2, 16, seed=2)
    with pytest.raises(ValueError):
        core.compute_losses(rae, dit, None, x, rae.encode(x).detach(), loss_type="image_l1",
                            noise_cfg=NOISE, lambda_fm=0.0,
                            y=torch.full((2,), 1000, dtype=torch.long),
                            corruption="flow", size=16, disc=head, disc_cfg=_DISC)


def test_n_reconstruct_with_disc_score():
    """score='disc' is sigmoid(head(z_in, z^)), upsampled to the image size."""

    class ZeroDiT(nn.Module):
        def forward(self, zt, t, y=None):
            return torch.zeros_like(zt)

    class ConstHead(nn.Module):
        def forward(self, za, r):
            return torch.zeros(za.shape[0], 1, *za.shape[-2:])

    rae = TinyRAE()
    x = _batch(2, 16, seed=22)
    _, maps, meta = I.reconstruct(rae, ZeroDiT(), x, [0, 1], t=0.1, corruption="simplenet",
                                  score="disc", disc=ConstHead())
    assert meta["score"] == "disc" and maps.shape == (2, 16, 16)
    assert torch.allclose(maps, torch.full_like(maps, 0.5))
    with pytest.raises(AssertionError):
        I.reconstruct(rae, ZeroDiT(), x, [0, 1], t=0.1, corruption="simplenet", score="disc")


# ---------------------------------------------------------------------------
# O. anchored adaptor for the decoder-free arm (experiment K)
# ---------------------------------------------------------------------------

_ANCHOR = {"type": "simplenet", "rank": 1, "preserve_norm": True, "inverse_head": True,
           "lambda_inv": 1.0}


def test_o_anchored_adaptor_starts_at_j_and_keeps_norms():
    rae, dit = TinyRAE(), TinyDiT()
    M.freeze_encoder(rae)
    M.freeze_decoder(rae)
    x = _batch(2, 32, seed=31)
    z0 = rae.encode(x).detach()
    ad = A.build_adaptor(_ANCHOR, dim=z0.shape[1])
    assert isinstance(ad, A.AnchoredAdaptor) and ad.is_anchored
    # identity at init: K's step 0 is exactly J's
    assert torch.equal(ad(z0), z0)
    assert float(ad.anchor_loss(z0, ad(z0))) == 0.0
    y = torch.full((2,), 1000, dtype=torch.long)
    kw = dict(loss_type="latent", noise_cfg={"sigma": 0.1, "mask": "perlin", "p_clean": 0.0},
              lambda_fm=0.0, y=y, corruption="simplenet", size=32)
    seed_all(4)
    _, t_j, _, _ = core.compute_losses(rae, dit, None, x, z0, **kw)
    seed_all(4)
    _, t_k, _, _ = core.compute_losses(rae, dit, None, x, z0, adaptor=ad, **kw)
    assert t_k["lat"] == pytest.approx(t_j["lat"], rel=1e-12, abs=1e-12)
    assert t_k["inv"] == 0.0

    # after the adaptor moves: token norms are still preserved, the anchor is
    # positive, and gradients reach both the adaptor and its inverse head
    with torch.no_grad():
        for p in ad.parameters():
            p.add_(0.3 * torch.randn_like(p))
    zt = ad(z0)
    assert torch.allclose(zt.norm(dim=1), z0.norm(dim=1), rtol=1e-5, atol=1e-6)
    ad.zero_grad(set_to_none=True)
    loss = ad.anchor_loss(z0, zt)
    assert float(loss) > 0
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for n, p in ad.named_parameters() if n.startswith("inverse."))
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for n, p in ad.named_parameters() if not n.startswith("inverse."))


def test_o_anchor_penalizes_collapse_and_guards():
    """A collapsed map (every token -> one vector) cannot be inverted: L_inv > 0."""
    torch.manual_seed(32)
    z = torch.randn(2, 3, 4, 4)
    ad = A.build_adaptor(_ANCHOR, dim=3)
    collapsed = torch.ones_like(z)  # all tokens identical
    assert float(ad.anchor_loss(z, collapsed)) > 0.1

    # plain trainable adaptor + latent loss is still refused; the anchored one is not
    rae, dit = TinyRAE(), TinyDiT()
    x = _batch(2, 16, seed=33)
    z0 = rae.encode(x).detach()
    kw = dict(loss_type="latent", noise_cfg={"sigma": 0.1, "mask": "all", "p_clean": 0.0},
              lambda_fm=0.0, y=torch.full((2,), 1000, dtype=torch.long),
              corruption="simplenet", size=16)
    with pytest.raises(ValueError):
        core.compute_losses(rae, dit, None, x, z0,
                            adaptor=A.build_adaptor({"type": "simplenet", "rank": 1}, dim=3), **kw)
    half = A.build_adaptor({**_ANCHOR, "inverse_head": False}, dim=3)
    assert not half.is_anchored
    with pytest.raises(ValueError):
        core.compute_losses(rae, dit, None, x, z0, adaptor=half, **kw)
    total, terms, _, _ = core.compute_losses(rae, dit, None, x, z0,
                                             adaptor=A.build_adaptor(_ANCHOR, dim=3), **kw)
    assert math.isfinite(float(total)) and "inv" in terms


def test_o_pixel_score_decodes_through_the_inverse_head():
    class ZeroDiT(nn.Module):
        def forward(self, zt, t, y=None):
            return torch.zeros_like(zt)

    rae = TinyRAE()
    x = _batch(2, 16, seed=34)
    ad = A.build_adaptor(_ANCHOR, dim=3)
    with torch.no_grad():
        for p in ad.inverse.parameters():
            p.add_(0.2 * torch.randn_like(p))
    ad.eval()
    xhat, _, _ = I.reconstruct(rae, ZeroDiT(), x, [0, 1], t=0.1, corruption="simplenet",
                               adaptor=ad)
    with torch.no_grad():
        expected = ad.invert(ad(rae.encode(x))).clamp(0, 1)  # z^ = z~ with zero velocity
    assert torch.allclose(xhat, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# P. noise range and EMA placement
# ---------------------------------------------------------------------------

def test_p_sigma_range_is_per_sample_and_midpoint_at_test():
    """sigma: [lo, hi] draws one sigma per sample; eval conditions on the midpoint."""
    sn = core.simplenet_cfg({"sigma": [0.5, 1.5]})
    assert sn["sigma_range"] == (0.5, 1.5) and sn["sigma"] == pytest.approx(1.0)
    assert core.simplenet_cfg({"sigma": "0.5,1.5"})["sigma_range"] == (0.5, 1.5)
    assert core.simplenet_cfg({"sigma": [0.2]})["sigma_range"] == (0.2, 0.2)
    assert core.simplenet_cfg({"sigma": 0.1})["sigma_range"] == (0.1, 0.1)

    seed_all(41)
    z0 = torch.randn(64, 8, 4, 4)
    s = core.sample_sigma(z0, {"sigma": [0.5, 1.5]})
    assert s.shape == (64,) and float(s.min()) >= 0.5 and float(s.max()) <= 1.5
    assert float(s.std()) > 0.1                         # actually varies
    const = core.sample_sigma(z0, {"sigma": 0.1})
    assert torch.equal(const, torch.full((64,), 0.1))

    # per-sample noise is exact: z_in - z0 == sigma_b * eps on every element
    m = torch.ones(64, 1, 4, 4)
    eps = torch.randn_like(z0)
    zt = core.simplenet_corrupt(z0, eps, s, m)
    assert torch.allclose(zt - z0, s.view(-1, 1, 1, 1) * eps, atol=1e-5)

    # oracle exactness still holds with per-sample t
    tc = core.conditioning_time(s, "simplenet").view(-1, 1, 1, 1)
    zhat = core.prediction_to_clean_latent(zt, (zt - z0) / tc, s, corruption="simplenet")
    assert torch.allclose(zhat, z0, atol=1e-5)


def test_p_latent_loss_with_sigma_range():
    """J's objective runs with a range and is reproducible under seed_all."""
    rae, dit = TinyRAE(), TinyDiT()
    x = _batch(2, 32, seed=43)
    z0 = rae.encode(x)
    y = torch.full((2,), 1000, dtype=torch.long)
    kw = dict(loss_type="latent", noise_cfg={"sigma": [0.5, 1.5], "mask": "perlin",
                                              "p_clean": 0.0, "w_anom": 5.0},
              lambda_fm=0.0, y=y, corruption="simplenet", size=32)
    seed_all(44)
    t1, terms1, _, _ = core.compute_losses(rae, dit, None, x, z0, **kw)
    seed_all(44)
    t2, _, _, _ = core.compute_losses(rae, dit, None, x, z0, **kw)
    assert math.isfinite(float(t1)) and "lat" in terms1
    assert torch.equal(t1, t2)


def test_p_validation_keeps_gpu_ema_in_place():
    """Only CPU-resident modules are moved; 'cuda:0' params are not 'different' from 'cuda'."""
    from iad import train as T

    cpu_mod = nn.Linear(2, 2)
    gpu_like = nn.Linear(2, 2, device="meta")   # stands in for a GPU-resident EMA
    assert T.modules_to_move([cpu_mod, gpu_like], torch.device("cuda")) == [cpu_mod]
    assert T.modules_to_move([cpu_mod, gpu_like], torch.device("cuda", 0)) == [cpu_mod]
    assert T.modules_to_move([cpu_mod], torch.device("cpu")) == []


# ---------------------------------------------------------------------------
# Q. structured synthetic anomalies (swap / dtd / mix)
# ---------------------------------------------------------------------------

def _q_noise(kind, **kw):
    cfg = {"sigma": [0.5, 1.5], "kind": kind, "mask": "perlin", "p_clean": 0.0,
           "w_anom": 5.0, "beta": [0.5, 1.0]}
    cfg.update(kw)
    return cfg


def test_q_swap_moves_real_tokens():
    """beta = 1 on a full mask: every sample becomes a shifted/flipped batch member."""
    seed_all(51)
    z0 = torch.randn(4, 3, 8, 8)
    m1, beta1 = torch.ones(4, 1, 8, 8), torch.ones(4)
    zs = core.latent_swap(z0, m1, beta1)
    for k in range(4):
        vals = torch.sort(zs[k].flatten()).values
        assert any(torch.allclose(vals, torch.sort(z0[j].flatten()).values) for j in range(4))
    # empty mask -> exact identity; partial beta -> a convex blend
    assert torch.equal(core.latent_swap(z0, torch.zeros(4, 1, 8, 8), beta1), z0)
    half = core.latent_swap(z0, m1, torch.full((4,), 0.5))
    assert torch.isfinite(half).all() and not torch.equal(half, z0)


def test_q_texture_paste_uses_the_bank():
    """beta = 1 on a full pixel mask gives a (possibly flipped) bank texture."""
    bank = torch.rand(5, 3, 16, 16).half()
    core.set_texture_bank(bank)
    try:
        seed_all(52)
        x = torch.rand(3, 3, 16, 16)
        out = core.texture_paste(x, torch.ones(3, 1, 16, 16), torch.ones(3))
        flips = [lambda t: t, lambda t: t.flip(-1), lambda t: t.flip(-2),
                 lambda t: t.flip(-1).flip(-2)]
        for k in range(3):
            assert any(torch.allclose(out[k], f(bank[i].float()), atol=1e-3)
                       for i in range(5) for f in flips)
        assert torch.equal(core.texture_paste(x, torch.zeros(3, 1, 16, 16), torch.ones(3)), x)
    finally:
        core.set_texture_bank(None)
    with pytest.raises(RuntimeError):
        core.texture_paste(x, torch.ones(3, 1, 16, 16), torch.ones(3))


def test_q_structured_corrupt_mix():
    """mix: one kind per sample, Gaussian samples carry their own sigma, the rest the
    test-time midpoint; dtd samples are E(pasted image); clean samples stay exact."""
    rae = TinyRAE()
    core.set_texture_bank(torch.rand(8, 3, 32, 32).half())
    try:
        x = _batch(16, 32, seed=53)
        z0 = rae.encode(x)
        seed_all(54)
        zin, m, t, kinds = core.structured_corrupt(
            z0, x, rae, None, _q_noise("mix", p_clean=0.25), 32)
        assert set(kinds) <= {"gaussian", "swap", "dtd"} and len(set(kinds)) >= 2
        assert zin.shape == z0.shape and m.shape == (16, 1, 32, 32)
        for k, kind in enumerate(kinds):
            if float(m[k].sum()) == 0:
                assert torch.equal(zin[k], z0[k])                    # p_clean
            elif kind == "gaussian":
                assert 0.5 <= float(t[k]) <= 1.5
            else:
                assert float(t[k]) == pytest.approx(1.0)
                # outside the mask nothing moves (swap) / only the paste changes (dtd)
                out = (m[k, 0] == 0)
                assert torch.allclose(zin[k][:, out], z0[k][:, out], atol=1e-6)
        assert core.needs_textures(_q_noise("mix")) and core.needs_textures(_q_noise("dtd"))
        assert not core.needs_textures(_q_noise("swap"))
        assert not core.needs_textures(_q_noise("mix", kinds=["gaussian", "swap"]))
    finally:
        core.set_texture_bank(None)


@pytest.mark.parametrize("kind", ["swap", "dtd", "mix"])
def test_q_latent_loss_runs_with_structured_corruption(kind):
    """J3 / J4 / J5 objectives: finite, reproducible, and gradients reach the DiT."""
    rae, dit = TinyRAE(), TinyDiT()
    core.set_texture_bank(torch.rand(6, 3, 32, 32).half())
    try:
        x = _batch(2, 32, seed=55)
        z0 = rae.encode(x).detach()
        y = torch.full((2,), 1000, dtype=torch.long)
        kw = dict(loss_type="latent", noise_cfg=_q_noise(kind), lambda_fm=0.0, y=y,
                  corruption="simplenet", size=32)
        seed_all(56)
        t1, terms, zt, _ = core.compute_losses(rae, dit, None, x, z0, **kw)
        seed_all(56)
        t2, _, _, _ = core.compute_losses(rae, dit, None, x, z0, **kw)
        assert math.isfinite(float(t1)) and "lat" in terms and torch.equal(t1, t2)
        t1.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dit.parameters())
    finally:
        core.set_texture_bank(None)


# ---------------------------------------------------------------------------
# R. per-category checkpoints
# ---------------------------------------------------------------------------

def test_r_slim_checkpoint(tmp_path):
    from iad import train as T

    net = nn.Linear(4, 4)
    opt = torch.optim.AdamW(net.parameters())
    net(torch.randn(2, 4)).sum().backward()
    opt.step()
    path = tmp_path / "carpet_best.pt"
    torch.save({"step": 7, "dit": net.state_dict(), "ema": net.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": {"x": 1}}, path)
    T.slim_checkpoint(path)
    st = T.load_checkpoint(path)
    assert st["slim"] and "optimizer" not in st and "scheduler" not in st
    assert st["step"] == 7
    assert all(torch.equal(st["dit"][k], v) for k, v in net.state_dict().items())
    assert not list(tmp_path.glob("*.tmp"))


def test_r_eval_only_checkpoint(tmp_path):
    """Per-category best checkpoints: EMA weights only, no optimizer/scheduler."""
    import copy
    from iad import train as T

    dit = nn.Linear(4, 4)
    ema = copy.deepcopy(dit)
    with torch.no_grad():
        ema.weight.add_(1.0)
    opt = torch.optim.AdamW(dit.parameters())
    path = tmp_path / "carpet_best.pt"
    T.save_checkpoint(path, 3, dit, ema, TinyRAE(), opt, None,
                      {"experiment": {}}, {}, eval_only=True)
    st = T.load_checkpoint(path)
    assert st["slim"] and st["model"] is None and st["optimizer"] is None
    assert torch.equal(st["ema"]["weight"], ema.weight)
    # without an EMA the raw weights are kept
    T.save_checkpoint(path, 3, dit, None, TinyRAE(), opt, None,
                      {"experiment": {}}, {}, eval_only=True)
    st = T.load_checkpoint(path)
    assert torch.equal(st["model"]["weight"], dit.weight) and st["optimizer"] is None


# ---------------------------------------------------------------------------
# S. EMA mirrors buffers (BatchNorm running statistics)
# ---------------------------------------------------------------------------

def test_s_ema_step_mirrors_batchnorm_buffers():
    """After training steps, EMA(adaptor).eval() must use the live running stats,
    not the initial (0, 1) ones, or F/G/K evaluate a different function."""
    import copy
    from iad import train as T

    seed_all(61)
    ad = A.build_adaptor({"type": "simplenet", "rank": 4, "norm": "batch"}, dim=16)
    ema = copy.deepcopy(ad)
    ema.requires_grad_(False)
    bn = ad.net[1]
    assert isinstance(bn, nn.BatchNorm1d)
    ad.train()
    for _ in range(5):                      # live BN stats move away from (0, 1)
        ad(torch.randn(2, 16, 4, 4) * 3 + 1)
        T.ema_step(ema, ad, 0.5)
    ema_bn = ema.net[1]
    assert not torch.allclose(bn.running_mean, torch.zeros(bn.num_features))
    assert torch.equal(ema_bn.running_mean, bn.running_mean)
    assert torch.equal(ema_bn.running_var, bn.running_var)
    assert torch.equal(ema_bn.num_batches_tracked, bn.num_batches_tracked)
    # parameters follow the decay rule, not a copy
    before = ema.net[0].weight.clone()
    with torch.no_grad():
        ad.net[0].weight.add_(1.0)
    T.ema_step(ema, ad, 0.5)
    assert torch.allclose(ema.net[0].weight, 0.5 * before + 0.5 * ad.net[0].weight)
    # the EMA in eval mode now applies BN with the live statistics
    ema.eval()
    z = torch.randn(1, 16, 4, 4)
    ref = ema.net[0](z.permute(0, 2, 3, 1).reshape(-1, 16))
    expect = (ref - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * ema_bn.weight + ema_bn.bias
    assert torch.allclose(ema_bn(ref), expect, atol=1e-6)


def test_s_ema_step_cpu_resident():
    """CPU-resident EMA (ema_device: cpu): same decay math, buffers mirrored."""
    import copy
    from iad import train as T

    net = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    ema = copy.deepcopy(net)
    net.train()
    net(torch.randn(8, 3) + 2)
    with torch.no_grad():
        net[0].weight.fill_(1.0)
        ema[0].weight.fill_(0.0)
    T.ema_step(ema, net, 0.9)
    assert torch.allclose(ema[0].weight, torch.full((3, 3), 0.1))
    assert torch.equal(ema[1].running_mean, net[1].running_mean)


# ---------------------------------------------------------------------------
# S. MVTec LOCO metrics in the eval scorer
# ---------------------------------------------------------------------------

def test_s_loco_score_setting():
    """Per-type image AUROC (each type vs good) and sPRO columns for LOCO rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rae_dit_iad_cli", Path(__file__).resolve().parent.parent / "rae_dit_iad.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)

    size = 32
    rng = np.random.default_rng(0)
    types = np.array(["good"] * 4 + ["logical_anomalies"] * 2 + ["structural_anomalies"] * 2)
    labels = (types != "good").astype(int)
    masks = np.zeros((8, size, size), bool)
    masks[4:, 8:16, 8:16] = True
    maps = rng.random((8, size, size)).astype(np.float32) * 0.1
    maps[6:, 8:16, 8:16] += 1.0            # only the structural defects are found
    maps[4:6] *= 0.5                       # logical images score below every good one
    defects = [(i, masks[i].copy(), 16.0) for i in range(4, 8)]
    row = R._score_setting(maps, masks, labels, size, 0.0,
                           loco=dict(types=types, defects=defects))
    assert row["img_auroc_max_structural"] == pytest.approx(1.0)
    assert row["img_auroc_max_logical"] == pytest.approx(0.0)
    assert row["spro_structural"] > 0.9 > row["spro_logical"]
    assert row["spro_logical"] < row["spro"] < row["spro_structural"]
    plain = R._score_setting(maps, masks, labels, size, 0.0)
    assert "spro" not in plain and "img_auroc_max_logical" not in plain


# ---------------------------------------------------------------------------
# T. localization ablation: score variants, pixel head, 32x32 grid
# ---------------------------------------------------------------------------

class _ShiftDiT(nn.Module):
    """Predicts v = c * z (per-token, translation-equivariant): z^ = z - t*c*z."""

    def forward(self, zt, t, y=None):
        return 0.3 * zt


def test_t_score_spec_parsing():
    sp = I.parse_score_spec("fuse_dec+s4+gf")
    assert sp["base"] == "fuse_dec" and sp["shifts"] == 4 and sp["gf"]
    assert sp["needs"] == ("latent_cos", "dec_diff")
    assert I.parse_score_spec("oracle16")["needs"] == ()
    for bad in ("nope", "latent_cos+s3", "disc+zz", "oracle16+s4", ""):
        with pytest.raises(ValueError):
            I.parse_score_spec(bad)
    # nested shift sets: s1 in s4 in s16
    s1, s4, s16 = (set(I.shift_offsets(k, 16)) for k in (1, 4, 16))
    assert s1 == {(0, 0)} and s1 <= s4 <= s16 and len(s16) == 16
    assert s4 == {(0, 0), (0, 8), (8, 0), (8, 8)}


def test_t_shift_unshift_roundtrip():
    x = torch.rand(2, 3, 32, 32)
    for dy, dx in ((0, 0), (4, 0), (0, 8), (8, 4)):
        xs = I.shift_image(x, dy, dx)
        assert xs.shape == x.shape
        assert torch.equal(xs[..., : 32 - dy, : 32 - dx], x[..., dy:, dx:])
        m = x[:, 0]
        back = I.unshift_map(I.shift_image(x, dy, dx)[:, 0], dy, dx)
        assert torch.equal(back[:, dy:, dx:], m[:, dy:, dx:])


def test_t_guided_filter_sharpens_a_token_map():
    """A bilinearly upsampled token map whose defect is visible in the image: the
    guided filter (radius = one token) snaps it to the image edges."""
    from sklearn.metrics import roc_auc_score
    S = 64
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    sharp = ((((yy - 30) ** 2 + (xx - 22) ** 2) < 8 ** 2)
             | ((yy > 50) & (yy < 52) & (xx > 10) & (xx < 55))).float()
    guide = (torch.rand(1, 3, S, S, generator=torch.Generator().manual_seed(0)) * 0.1
             + 0.8 * sharp)
    tok = F.adaptive_avg_pool2d(sharp[None, None], 8)
    p = F.interpolate(tok, size=(S, S), mode="bilinear", align_corners=False)[:, 0]
    out = I.guided_filter(p, guide, r=8)
    assert out.shape == p.shape
    lab = sharp.ravel() > 0
    assert roc_auc_score(lab, out.ravel()) > roc_auc_score(lab, p.ravel()) + 0.002
    const = torch.full((1, S, S), 0.7)
    assert torch.allclose(I.guided_filter(const, torch.rand(1, 3, S, S), r=3), const, atol=1e-5)


def test_t_mask_oracle_is_grid_limited():
    from sklearn.metrics import roc_auc_score
    masks = np.zeros((2, 64, 64), bool)
    masks[0, 10:12, 5:60] = True                             # thin scratch
    o16, o64 = I.mask_oracle(masks, 16), I.mask_oracle(masks, 64)
    a16 = roc_auc_score(masks.ravel(), o16.ravel())
    a64 = roc_auc_score(masks.ravel(), o64.ravel())
    assert a64 > 0.999 and a16 < a64                         # coarser grid, worse ceiling


def test_t_score_maps_bases_and_fusion():
    rae, dit = TinyRAE(), _ShiftDiT()
    x = _batch(2, 32, seed=71)
    specs = [I.parse_score_spec(s) for s in
             ("latent_cos", "pixel", "dec_diff", "fuse_dec", "fuse_pixel", "latent_cos+s4")]
    xh, maps = I.score_maps(rae, dit, x, specs, t=1.0, patch_px=8)  # s4: 4 px offsets
    assert set(maps) == {s["spec"] for s in specs}
    for m in maps.values():
        assert m.shape == (2, 32, 32) and torch.isfinite(m).all()
    # same numbers as the single-score path
    _, ref, _ = I.reconstruct(rae, dit, x, [0, 1], t=1.0, corruption="simplenet",
                              score="latent_cos")
    assert torch.allclose(maps["latent_cos"], ref, atol=1e-6)
    _, refp, _ = I.reconstruct(rae, dit, x, [0, 1], t=1.0, corruption="simplenet",
                               score="pixel")
    assert torch.allclose(maps["pixel"], refp, atol=1e-6)
    # identity decoder: D(z) = x, so the decoder difference equals the pixel score
    assert torch.allclose(maps["dec_diff"], maps["pixel"], atol=1e-6)
    assert torch.allclose(maps["fuse_dec"], maps["latent_cos"] * maps["dec_diff"], atol=1e-6)
    # a translation-equivariant model: shift TTA leaves the interior unchanged
    assert torch.allclose(maps["latent_cos+s4"][:, 4:-4, 4:-4],
                          maps["latent_cos"][:, 4:-4, 4:-4], atol=1e-5)
    assert not torch.allclose(maps["latent_cos+s4"], maps["latent_cos"])  # borders differ
    with pytest.raises(ValueError):
        I.score_maps(rae, dit, x, [I.parse_score_spec("disc")], t=1.0)


_PIXDISC = {"enabled": True, "d": 8, "blocks": 1, "pixel": True, "pixel_ch": 8}


def test_t_pixel_head_starts_at_token_map_and_keeps_token_head():
    from iad import disc as DS
    plain = DS.build_discriminator({"enabled": True, "d": 8, "blocks": 1}, dim=3, seed=4)
    both = DS.build_discriminator(_PIXDISC, dim=3, seed=4)
    for a, b in zip(plain.parameters(), both.seg.parameters()):
        assert torch.equal(a, b)                             # token head init unchanged
    assert len(DS.head_parts(both)) == 2 and len(DS.head_parts(plain)) == 1
    x = _batch(2, 32, seed=72)
    za, r = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    logit = both(za, r)
    assert torch.equal(logit, plain(za, r))
    px = both.pixel(x, za, r, logit)
    up = F.interpolate(logit, size=(32, 32), mode="bilinear", align_corners=False)
    assert px.shape == (2, 1, 32, 32) and torch.allclose(px, up, atol=1e-6)
    with pytest.raises(ValueError):
        DS.disc_cfg({"pixel": True, "pixel_ch": 12})
    # clipping: a part with no gradient this step still combines (job 163739)
    assert DS.clip_head(None, 1.0) is None
    both.zero_grad(set_to_none=True)
    both(za, r).sum().backward()                     # token head only: pix has no grads
    n = DS.clip_head(both, 1.0)
    assert n.device.type == "cpu" and torch.isfinite(n) and float(n) > 0
    assert all(p.grad is None for p in both.pix.parameters())


def test_t_pixel_head_loss_leaves_dit_and_token_head_unchanged():
    from iad import disc as DS
    rae, dit = TinyRAE(), TinyDiT()
    core.set_texture_bank(torch.rand(6, 3, 32, 32).half())
    try:
        x = _batch(4, 32, seed=73)
        z0 = rae.encode(x).detach()
        y = torch.full((4,), 1000, dtype=torch.long)
        noise = _q_noise("dtd", p_clean=0.25)
        kw = dict(loss_type="latent", noise_cfg=noise, lambda_fm=0.0, y=y,
                  corruption="simplenet", size=32)
        tok_cfg = {k: v for k, v in _PIXDISC.items() if not k.startswith("pixel")}
        runs = []
        for cfg in (tok_cfg, _PIXDISC):
            head = DS.build_discriminator(cfg, dim=3, seed=0)
            for p in dit.parameters():
                p.grad = None
            seed_all(74)
            tot, terms, _, _ = core.compute_losses(rae, dit, None, x, z0, disc=head,
                                                   disc_cfg=cfg, **kw)
            tot.backward()
            seg = head.seg if hasattr(head, "seg") else head
            runs.append((terms, [p.grad.clone() for p in dit.parameters()],
                         [p.grad.clone() for p in seg.parameters()], head))
        (t0, g0, s0, _), (t1, g1, s1, h1) = runs
        assert "pix" in t1 and math.isfinite(t1["pix"]) and "pix" not in t0
        assert t1["seg"] == pytest.approx(t0["seg"], rel=1e-6)
        for a, b in zip(g0 + s0, g1 + s1):
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-8)
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in h1.pix.parameters())
        # Gaussian-only corruption has no image-space anomaly: refused
        with pytest.raises(ValueError):
            core.compute_losses(rae, dit, None, x, z0, disc=h1, disc_cfg=_PIXDISC,
                                **{**kw, "noise_cfg": _q_noise("swap")})
    finally:
        core.set_texture_bank(None)


def test_t_score_maps_pxhead():
    from iad import disc as DS
    rae, dit = TinyRAE(), _ShiftDiT()
    head = DS.build_discriminator(_PIXDISC, dim=3, seed=0).eval()
    x = _batch(2, 32, seed=75)
    _, maps = I.score_maps(rae, dit, x, [I.parse_score_spec("pxhead"),
                                         I.parse_score_spec("disc")], t=1.0, disc=head)
    # zero-initialized residual: the pixel head equals the upsampled token map
    assert torch.allclose(maps["pxhead"], maps["disc"], atol=1e-5)
    plain = DS.build_discriminator({"enabled": True, "d": 8, "blocks": 1}, dim=3, seed=0)
    with pytest.raises(ValueError):
        I.score_maps(rae, dit, x, [I.parse_score_spec("pxhead")], t=1.0, disc=plain)


def test_t_regrid_stats_and_pooled_decode():
    st = torch.randn(1, 4, 2, 2)
    assert M._regrid_stat(st, 4).shape == (1, 4, 4, 4)
    assert M._regrid_stat(torch.randn(4, 1, 1), 4).shape == (4, 1, 1)   # per-channel: kept

    class FakeRAE(nn.Module):
        def __init__(self):
            super().__init__()
            self._dec_grid = 2
            self.seen = None

        def decode(self, z):
            self.seen = z
            return z

    r = FakeRAE()
    z = torch.arange(16.).view(1, 1, 4, 4)
    out = M.decode_latent(r, z)
    assert out.shape == (1, 1, 2, 2)
    assert torch.allclose(out, F.adaptive_avg_pool2d(z, 2))
    assert M.decode_latent(r, out).shape == (1, 1, 2, 2)


def test_t_regrid_official_dit():
    """The released DiT-DH architecture runs on a 2x finer grid after re-gridding."""
    pytest.importorskip("stage2.models.DDT")
    pytest.importorskip("timm")
    from stage2.models.DDT import DiTwDDTHead
    torch.manual_seed(0)
    dit = DiTwDDTHead(input_size=4, patch_size=1, in_channels=8, hidden_size=[32, 32],
                      depth=[1, 1], num_heads=[2, 2], num_classes=10).eval()
    pe = dit.pos_embed.detach().clone()
    M.regrid_dit(dit, 8)
    assert dit.pos_embed.shape == (1, 64, 32) and not dit.pos_embed.requires_grad
    assert dit.enc_feat_rope.freqs_cos.shape[0] == 64
    z = torch.randn(2, 8, 8, 8)
    with torch.no_grad():
        v = dit(z, torch.rand(2), y=torch.zeros(2, dtype=torch.long))
    assert v.shape == z.shape and torch.isfinite(v).all()
    # the state dict round-trips at the new grid (eval rebuilds, re-grids, then loads)
    dit2 = DiTwDDTHead(input_size=4, patch_size=1, in_channels=8, hidden_size=[32, 32],
                       depth=[1, 1], num_heads=[2, 2], num_classes=10)
    M.regrid_dit(dit2, 8)
    dit2.load_state_dict(dit.state_dict(), strict=True)
    # re-gridding to the same size is a no-op
    dit3 = DiTwDDTHead(input_size=4, patch_size=1, in_channels=8, hidden_size=[32, 32],
                       depth=[1, 1], num_heads=[2, 2], num_classes=10)
    dit3.pos_embed.data.copy_(pe)
    M.regrid_dit(dit3, 4)
    assert torch.equal(dit3.pos_embed, pe)


def test_t_eval_multi_rows_on_a_synthetic_category(tmp_path):
    """The eval path for score variants: one row per (spec, blur), resumable,
    metrics at eval_size, the oracle beats the model, skipped specs reported."""
    import argparse
    import importlib.util
    from PIL import Image
    spec = importlib.util.spec_from_file_location(
        "rae_dit_iad_cli", Path(__file__).resolve().parent.parent / "rae_dit_iad.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)

    c = tmp_path / "carpet"
    rng = np.random.default_rng(0)
    def png(p, a):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(a.astype(np.uint8)).save(p)
    for i in range(2):
        png(c / "train" / "good" / f"{i:03d}.png", rng.integers(0, 255, (48, 48, 3)))
    for i in range(3):
        png(c / "test" / "good" / f"{i:03d}.png", rng.integers(0, 255, (48, 48, 3)))
    for i in range(2):
        png(c / "test" / "hole" / f"{i:03d}.png", rng.integers(0, 255, (48, 48, 3)))
        m = np.zeros((48, 48)); m[10:20, 10:30] = 255
        png(c / "ground_truth" / "hole" / f"{i:03d}_mask.png", m)

    args = argparse.Namespace(mvtec_root=str(tmp_path), batch_size=3, blur_sweep="0,2",
                              save_maps=None)
    cfg = {"size": 32, "eval_size": 24}
    done = set()
    kw = dict(cat="carpet", ckpt_path=Path("carpet_best.pt"), used_ema=True, exp_id="T",
              seed=0, t=1.0, num_classes=10, device=torch.device("cpu"), blur_sigma=2.0,
              done=done)
    specs = ["latent_cos", "fuse_dec+s4", "oracle16", "oracle16+gf", "disc"]
    rows, pi = R._eval_multi(TinyRAE(), _ShiftDiT(), None, None, cfg, args, specs, **kw)
    assert len(rows) == 4 * 2                                  # disc skipped (no head)
    assert {(r["score"], r["blur_sigma"]) for r in rows} == {
        (s, b) for s in specs[:4] for b in (0.0, 2.0)}
    assert len(pi) == 4 * 5 and all("score" in r for r in pi)
    orc = [r for r in rows if r["score"] == "oracle16" and r["blur_sigma"] == 0.0][0]
    assert orc["px_auroc"] > 0.95 and orc["img_auroc_max"] == pytest.approx(1.0)
    assert orc["dit_passes_per_img"] == 0
    # resumable: a second call with the same `done` adds nothing
    rows2, _ = R._eval_multi(TinyRAE(), _ShiftDiT(), None, None, cfg, args, specs, **kw)
    assert rows2 == []
    assert R._resize_maps(np.zeros((2, 32, 32), np.float32), 24).shape == (2, 24, 24)


# ---------------------------------------------------------------------------
# U. exact DRAEM: SSIM, the image-space U-Net, loss='draem', score 'draem'
# ---------------------------------------------------------------------------

_UNET = {"enabled": True, "type": "unet", "unet_base": 8, "detach": False}


def test_u_ssim_and_reconstruction_loss():
    x = _batch(2, 32, seed=81)
    assert float(core.ssim(x, x)) == pytest.approx(1.0, abs=1e-5)
    noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    assert float(core.ssim(x, noisy)) < 0.9
    tot, l2, ss = core.draem_reconstruction_loss(x, x)
    assert float(tot) == pytest.approx(0.0, abs=1e-5)
    tot, l2, ss = core.draem_reconstruction_loss(noisy, x)
    assert float(l2) > 0 and float(ss) > 0 and torch.allclose(tot, l2 + ss)


def test_u_unet_shapes_and_guards():
    from iad import disc as DS
    head = DS.build_discriminator(_UNET, dim=3, seed=0)
    assert DS.is_unet(head) and not DS.is_unet(None)
    assert len(DS.head_parts(head)) == 1
    x, r = _batch(2, 32, seed=82), _batch(2, 32, seed=83)
    assert head(x, r).shape == (2, 1, 32, 32)
    with pytest.raises(ValueError):
        head(_batch(1, 48), _batch(1, 48))                     # side not divisible by 32
    with pytest.raises(ValueError):
        DS.disc_cfg({"type": "unet", "pixel": True})
    with pytest.raises(ValueError):
        DS.disc_cfg({"type": "patch"})
    again = DS.build_discriminator(_UNET, dim=3, seed=0)        # forked RNG: reproducible
    assert all(torch.equal(a, b) for a, b in zip(head.parameters(), again.parameters()))


def test_u_draem_loss_trains_both_networks_and_guards():
    from iad import disc as DS
    rae, dit = TinyRAE(), TinyDiT()
    core.set_texture_bank(torch.rand(6, 3, 32, 32).half())
    try:
        x = _batch(4, 32, seed=84)
        z0 = rae.encode(x).detach()
        y = torch.full((4,), 1000, dtype=torch.long)
        kw = dict(loss_type="draem", noise_cfg=_q_noise("dtd", p_clean=0.25, beta=[0.2, 1.0]),
                  lambda_fm=0.0, y=y, corruption="simplenet", size=32)
        grads = {}
        for name, cfg in (("none", None), ("joint", _UNET), ("detached", {**_UNET, "detach": True})):
            head = DS.build_discriminator(cfg, dim=3, seed=0) if cfg else None
            for p in dit.parameters():
                p.grad = None
            seed_all(85)
            tot, terms, _, _ = core.compute_losses(rae, dit, None, x, z0, disc=head,
                                                   disc_cfg=cfg, **kw)
            assert math.isfinite(float(tot)) and {"img", "ssim", "total"} <= set(terms)
            if head is not None:
                assert "seg" in terms and float(tot) > terms["total"]   # seg kept out of total
            tot.backward()
            grads[name] = [p.grad.clone() for p in dit.parameters()]
            if head is not None:
                assert any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in head.parameters())
        assert any(g.abs().sum() > 0 for g in grads["none"])
        # detached: the U-Net never changes the DiT's gradients; joint (DRAEM): it does
        assert all(torch.allclose(a, b) for a, b in zip(grads["none"], grads["detached"]))
        assert not all(torch.allclose(a, b) for a, b in zip(grads["none"], grads["joint"]))
        # + latent term (DR_draem_lat)
        seed_all(85)
        _, terms, _, _ = core.compute_losses(rae, dit, None, x, z0, lambda_latent=1.0, **kw)
        assert "lat" in terms
        with pytest.raises(ValueError):                       # anomaly must be in the image
            core.compute_losses(rae, dit, None, x, z0, **{**kw, "noise_cfg": _q_noise("mix")})
        head = DS.build_discriminator(_UNET, dim=3, seed=0)
        with pytest.raises(ValueError):                       # the U-Net reads D(z^)
            core.compute_losses(rae, dit, None, x, z0, disc=head, disc_cfg=_UNET,
                                **{**kw, "loss_type": "latent"})
    finally:
        core.set_texture_bank(None)


def test_u_score_maps_draem():
    from iad import disc as DS
    rae, dit = TinyRAE(), TinyDiT()
    head = DS.build_discriminator(_UNET, dim=3, seed=0).eval()
    x = _batch(2, 32, seed=86)
    specs = [I.parse_score_spec(s) for s in ("draem", "draem+gf", "latent_cos", "pixel")]
    xh, maps = I.score_maps(rae, dit, x, specs, t=1.0, disc=head, patch_px=8)
    assert set(maps) == {s["spec"] for s in specs}
    for m in maps.values():
        assert m.shape == (2, 32, 32) and torch.isfinite(m).all()
    assert float(maps["draem"].min()) >= 0 and float(maps["draem"].max()) <= 1
    ref = torch.sigmoid(head(x, xh))[:, 0]
    assert torch.allclose(maps["draem"], ref, atol=1e-5)
    with pytest.raises(ValueError):                            # no U-Net
        I.score_maps(rae, dit, x, [I.parse_score_spec("draem")], t=1.0)
    with pytest.raises(ValueError):                            # token score on a U-Net
        I.score_maps(rae, dit, x, [I.parse_score_spec("disc")], t=1.0, disc=head)


@pytest.mark.parametrize("name", ["DR_draem", "DR_draem_lat", "DR_draem_dec", "DR_draem_lat_dec"])
def test_u_draem_configs_parse(name):
    from iad import disc as DS
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(
        str(Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml")))
    assert cfg["experiment"]["id"] == name and cfg["experiment"]["loss"] == "draem"
    dc = DS.disc_cfg(cfg["discriminator"])
    assert dc["type"] == "unet" and dc["enabled"] and not dc["detach"]
    sn = core.simplenet_cfg(cfg["noise"])
    assert sn["kind"] == "dtd" and sn["p_clean"] == 0.5
    assert cfg["inference"]["score"] == "draem"
    lat = name.startswith("DR_draem_lat")
    assert float(cfg["experiment"]["lambda_latent"]) == (1.0 if lat else 0.0)
    assert bool(cfg["experiment"]["decoder_finetune"]) == name.endswith("_dec")
