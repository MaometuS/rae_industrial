# RAE-DiT for Industrial Anomaly Detection (IAD)

Trains the **original RAE DiT** (DiT-DH-XL, `bytetriper/RAE`) on MVTec AD
*normal* images as a **latent restorer** and evaluates reconstruction-based
anomaly detection. The default corruption is **SimpleNet-style** (small Gaussian
noise on a Perlin-masked subset of tokens); experiment A keeps the original
flow-matching objective on the original flow path as the single reference.

Nothing in the official RAE repo is modified; it is consumed as a library
(`../rae_fidelity/RAE`, commit `a4d18c4`), and MVTec loading / benchmark
metrics come from `../latent_draem/latent_draem.py`.

---

## 1. Experiments

| id | config | corruption | loss | decoder | compare with |
|----|--------|-----------|------|---------|--------------|
| **A** | `configs/A_flow_matching.yaml` | **flow path** | official flow matching (untouched `transport.training_losses`) | unused in training | reference: the original RAE objective; **A vs J** isolates the corruption |
| **B** | `configs/B_image_l1.yaml` | SimpleNet | image L1 through the decoder | frozen | the hinge: C–G and J each change one thing |
| **C** | `configs/C_image_l1_decfinetune.yaml` | SimpleNet | image L1 | **fine-tuned** (0.1 × lr) | B (decoder) |
| **D** | `configs/D_image_l1_latent.yaml` | SimpleNet | image L1 + λ·latent MSE (λ = 1) | frozen | B (latent term) |
| **E** | `configs/E_image_l1_latent_decfinetune.yaml` | SimpleNet | image L1 + λ·latent MSE | **fine-tuned** | D (decoder), C (latent term) |
| **F** | `configs/F_image_l1_adaptor.yaml` | SimpleNet | image L1 + **feature adaptor** | frozen | B (domain gap) |
| **G** | `configs/G_image_l1_adaptor_decfinetune.yaml` | SimpleNet | image L1 + **feature adaptor** | **fine-tuned** | C (domain gap) |
| **J** | `configs/J_latent_simplenet_nodecoder.yaml` | SimpleNet | **latent MSE only** | **dropped** | D (decoder), A (corruption), latent_draem restorer, kNN |
| **K** | `configs/K_latent_simplenet_adaptor.yaml` | SimpleNet | latent MSE + **anchored feature adaptor** | **dropped** | J (adaptor), F (adaptor with a decoder) |
| **J2** | `configs/J2_latent_sigma05_15.yaml` | SimpleNet, **σ ~ U(0.5, 1.5)** | latent MSE | **dropped** | J (noise strength), A (flow path at the same test t_c = 0.5) |
| **K2** | `configs/K2_latent_adaptor_sigma05_15.yaml` | SimpleNet, **σ ~ U(0.5, 1.5)** | latent MSE + anchored adaptor | **dropped** | K, J2 |
| **J3** | `configs/J3_latent_swap.yaml` | **token swap**: shifted/flipped tokens of a batch member | latent MSE | **dropped** | J2 (Gaussian) |
| **J4** | `configs/J4_latent_dtd.yaml` | **DTD texture** pasted in pixel space, re-encoded (DRAEM) | latent MSE | **dropped** | J2 |
| **J5** | `configs/J5_latent_mix.yaml` | **mix**: Gaussian / swap / DTD per sample | latent MSE | **dropped** | J2, J3, J4 |
| **K5** | `configs/K5_latent_adaptor_mix.yaml` | mix, as J5 | latent MSE + anchored adaptor | **dropped** | J5, K2 |

Letters H and I were merged away: with SimpleNet as the default, "B with SimpleNet
noise" is B, and "image L1 + latent term" is D. The old flow-path hybrid
(L1 + 0.1·flow matching) is gone: on the SimpleNet path the flow-matching target
reduces to the latent MSE (see *Corruption* below), which D/E now use directly.

**Every arm is scored three ways** by the runner (`SCORES="pixel latent_cos disc"`): the
pixel map `|x − D(ẑ)|` after the decoder, the latent map `1 − cos(E(x), ẑ)` before it, and
DRAEM's learned segmentation map (`disc`, every arm but A — see *Discriminator* below).

Shared: encoder **always frozen** (and latents pre-cached — exact, since the
encoder never updates), official AdamW hyperparameters (lr 2e-4, betas
(0.9, 0.95), wd 0), official linear warmup/decay scheduler, official EMA
(0.9995), grad clip 1.0, class conditioning via the official CFG **null label**
(1000) because MVTec has no ImageNet label (a no-op w.r.t. the official label
dropout).

Noise ablations (train-time corruption, all on top of B):

| config | change vs B |
|--------|-------------|
| `noise_sigma005.yaml` | σ = 0.05 (default 0.1) |
| `noise_sigma02.yaml` | σ = 0.2 |
| `noise_mask_all.yaml` | `mask: all` — every token noised (SimpleNet exactly; no clean context) |

`SIGMA=<value>` overrides σ for any arm but A without a new config.

### Feature adaptor: is ImageNet→MVTec feature adaptation worth it? (F/G)

The encoder is frozen, so its latents carry an ImageNet inductive bias. F/G
insert a small trainable **feature adaptor** `A` between the frozen encoder and
the DiT/decoder; `A` defines the latent space both operate on:

```
x --encoder--> z0 --A--> z~0 --corruption--> z_t --DiT--> z^~ --decoder--> x^
```

Training and inference share that space, so no inverse of `A` is ever needed.
`A` is a **residual bottleneck with a zero-initialized head**
(`A(z) = z + h(z)`, `h ≡ 0` at init), hence *experiment F starts exactly at B*
(baseline loss equality is unit-tested) and the frozen decoder in F stays
in-distribution while `A` learns the domain correction; in G the decoder
co-adapts at `0.1 ×` DiT lr. Variants (`adaptor.type`):

| type | module | init | notes |
|------|--------|------|-------|
| `simplenet` (default) | `Linear → BatchNorm → LeakyReLU → Linear` per token | identity | SimpleNet's Feature Adaptor (CVPR 2023) |
| `mlp` | `LayerNorm → Linear → GELU → Linear` | identity | feature-translator style |
| `linear` / `conv1x1` | single linear map / 1×1 conv | identity | learned rotation/scale |
| `affine` | per-channel scale + bias | identity **or** `train_stats` | `train_stats` = `(z − μ)/σ` with μ,σ fit on the MVTec **train split** — the training-free statistic alignment of PatchCore/PaDiM/FeaNorm-style pipelines (`trainable: false` keeps it purely statistical) |

Note: PatchCore itself learns no adaptor (it adapts by greedy-coreset
subsampling of nominal features + nearest-neighbour scoring); that mechanism is
not applicable to a decoder-based pipeline, so `affine` + `train_stats` is the
closest analogue. Other knobs: `rank` (bottleneck = `latent_dim // rank`),
`norm` (`batch|layer|none`), `leaky_slope`, `dropout`, `trainable`;
`--adaptor <type>` / `--adaptor-rank` override per run and `ADAPTOR=mlp ... bash
run_dit_iad.sh` (or any preset) does the same for the whole matrix. The adaptor
is EMA'd with the same decay as the DiT and stored in the checkpoints, so `eval`
reconstructs it automatically from the checkpoint's own config.

### Corruption: SimpleNet-style by default, flow path only in A

**Flow path (A only).** The original RAE corruption mixes noise into *every*
token and shrinks the clean signal: `z_t = (1−t)·z₀ + t·ε`, with t drawn from the
official warped logit-normal (median t ≈ 0.87, only ~2.6% of draws below 0.5). The
DiT learns the velocity `v* = ε − z₀` along that straight data→noise line, i.e.
to *generate* a whole latent from mostly noise. At test time A corrupts at t = 0.5
with per-image seeded ε and jumps back in one step, `ẑ = z_t − t·v`.

**SimpleNet-style (default, B–J; `corruption: simplenet`, `iad/core.py`).** Small
i.i.d. Gaussian noise (SimpleNet's idea) on a DRAEM-style Perlin token mask only:

```
train:  m  = Perlin blob at 256 px, avg-pooled to 16×16   (p_clean = 0.2 → m = 0)
        z_in = z₀ + m · σ · ε,        σ = 0.1 (normalized-latent units)
        ẑ   = z_in − t_c · v_θ(z_in, t_c),   t_c = σ / (1 + σ)
        B/C/F/G:  L = ‖D(ẑ) − x‖₁ / 3HW
        D/E:      L = ‖D(ẑ) − x‖₁ / 3HW + λ · Σ_p w_p‖ẑ_p − z₀_p‖² / Σ_p w_p,   w = 1 + 4m,  λ = 1
        J:        L = Σ_p w_p‖ẑ_p − z₀_p‖² / Σ_p w_p                          (no decoder)
test:   z_in = E(x)  (no noise: the real defect is the corruption); mode = single
```

* Untouched tokens stay exactly clean and the signal is not shrunk, so the DiT must
  judge corrupted tokens against their clean context ("replace what does not
  fit"): a *restorer*, not a generator — the mechanism of `latent_draem`'s
  `latent_noise` restorer.
* σ is small so corrupted tokens cannot be spotted by norm alone (σ = 0.1 adds ~1%
  to a token's squared norm; `latent_noise` used σ ∈ [0.5, 1.5]). SimpleNet's own
  σ = 0.015 lives in its adapted WideResNet feature scale and does not transfer
  1:1, hence the σ ablations.
* **Why flow matching and the hybrid do not exist on this path.** There is no
  data→noise line, so "velocity along the path" is undefined. The output that makes
  `ẑ = z₀` exactly is `v* = (z_in − z₀)/t_c`, and `‖v − v*‖² = ‖ẑ − z₀‖² / t_c²`:
  flow matching collapses to the latent MSE (J's loss, D/E's extra term). The code
  therefore refuses `flow_matching`/`hybrid` with `corruption: simplenet`.
* **A vs J.** A's flow-matching loss is also a latent-space loss
  (`‖v − v*‖² = ‖ẑ − z₀‖² / t²` on the flow path) and A never trains through the
  decoder, so A and J differ essentially in the corruption. Compare them on the
  `latent_cos` score.
* `ode`/`multi` do not apply to SimpleNet arms (no test-time noise); eval uses the
  checkpoint's σ unless `--t` is given.

### Decoder-free: the DiT as a latent restorer (J) and latent scoring

The DINOv2-B RAE decoder reconstructs normal MVTec images at PSNR ≈ 20 / SSIM ≈ 0.56
(`rae_fidelity`), so every pixel-space loss and map carries decoder error. J drops
the decoder from the loss (`loss: latent`) and is scored in latent space:

```
test:   z = E(x), ẑ = z − t_c·v_θ(z, t_c)
        map_p = 1 − cos(z_p, ẑ_p)   (latent_l2: mean_c(ẑ_p − z_p)²)
        16×16 token map → bilinear 256 px → Gaussian blur σ = 4 → usual metrics
```

* The decoder is offloaded to CPU while J trains and only used at eval (pixel score,
  viz grids).
* J is `latent_draem`'s `latent_noise` restorer + residual score with the
  pretrained DiT as the restorer. Bars: that restorer (0.972 image AUROC, one seed)
  and frozen-token kNN (0.969).
* Latent scoring is an eval option for **every** checkpoint (`--score latent_cos`);
  the runner scores all arms with `pixel` and `latent_cos`. Rows carry a `score`
  column; rows from before it count as `pixel`.
* A *plain* trainable adaptor is refused with `loss: latent`: nothing anchors the
  adapted space, so it could shrink it, inflate it past the noise, or map every token
  to one vector, and win the loss. `decoder_finetune` must be false.

### K: the decoder-free arm with an anchored feature adaptor

K is J plus the SimpleNet feature adaptor, made collapse-proof
(`iad.adaptor.AnchoredAdaptor`, `preserve_norm` + `inverse_head`):

```
z~   = A(z₀) · ‖z₀‖ / ‖A(z₀)‖            per token: norm preserved (no shrink / inflate)
z_in = z~ + m·σ·ε,   ẑ = z_in − t_c·v_θ(z_in, t_c)
L    = Σ_p w_p ‖ẑ_p − z~_p‖² / Σ_p w_p  +  λ_inv · mean ‖B(z~) − z₀‖²     λ_inv = 1
test: latent_cos 1 − cos(z~, ẑ) · disc head · pixel |x − D(B(ẑ))|
```

* `B` is a second residual bottleneck that must rebuild the encoder latent from the
  adapted one, so the adaptor may re-express the space but not discard information;
  it also maps restorations back to the decoder's space for the pixel score.
* `A` and `B` are zero-initialized residuals: step 0 of K is exactly J (unit test O).
* Remaining loophole: an invertible map can still squash some directions to tiny
  magnitudes (with `B` scaling them back), which both the restoration loss and the
  cosine score would then under-weight. The rank-192 bottleneck limits this; watch
  `loss_inv` and per-channel spread of `z~` if K behaves oddly.
* Resolution limit: latent maps have 16×16 token resolution, so defects smaller
  than a 16-px patch (screw) are averaged into their token.

### Discriminator: DRAEM's discriminative sub-network (every SimpleNet arm)

DRAEM trains two networks: the reconstructive one (here the DiT) and a
*discriminative* segmentation network that sees the corrupted input and the
reconstruction together and predicts where the synthetic anomaly was pasted. At test
time **its output, not the raw residual, is the anomaly map**: it learns which
differences are defects and which are ordinary reconstruction error. B–G, J and the
ablations carry it (`discriminator:` block, `iad/disc.py`), on the token grid:

```
logits = D([z_in, ẑ, z_in − ẑ, 1 − cos(z_in, ẑ)])     (latent_draem.Discriminator)
L_seg  = focal_bce(logits, m)      m = the SimpleNet token mask (soft coverage), γ = 2
score  = sigmoid(logits) → bilinear 256 px → blur σ = 4 → usual metrics   (--score disc)
```

* **It does not change any arm.** With `detach: true` (default) the head gets
  detached copies of `z_in` and `ẑ`, its own gradient clipping and an RNG-forked
  initialization, so the DiT's gradients are identical with and without it (unit
  test N). Its loss is logged as `seg` and kept out of `total`, so it cannot change
  which DiT checkpoint is selected as `best`. Every arm comparison stays valid; the
  head only adds a learned score on top. `detach: false` trains it jointly (DRAEM's
  own setting), which does change the DiT.
* EMA'd like the DiT and stored in the checkpoints; eval pairs the EMA head with the
  EMA DiT, exactly as validation does.
* A (flow path) has no synthetic-anomaly mask, so no head: `--score disc` is skipped
  for A with a message. In `noise_mask_all` the mask covers every token, so the head
  can only learn "this sample was corrupted" — expect it to be weak there.
* Resolution: token-level, like the latent scores. A pixel-level head on (image,
  decoded restoration) would be the full-resolution DRAEM variant.

## 2. Layout

```
rae_dit_iad/
├── iad/
│   ├── core.py       # corruption, time sampling, velocity→clean conversion, losses
│   ├── models.py     # official RAE/DiT/transport builders, freezing, optimizer
│   ├── adaptor.py    # feature adaptors after the frozen encoder (experiments F/G)
│   ├── disc.py       # DRAEM discriminative head on the token grid (SimpleNet arms)
│   ├── train.py      # training engine (EMA, official scheduler, checkpoint/resume)
│   └── infer.py      # single/multi/ode reconstruction, anomaly maps, viz grids
├── rae_dit_iad.py    # CLI: `train` and `eval` subcommands
├── configs/          # experiment + noise-ablation YAMLs
├── tests/test_iad.py # Tests A–I (+J memory, +K adaptor, +L SimpleNet, +M decoder-free, +N disc, +O anchor)
├── run_dit_iad.sh    # cluster runner (shared train + eval matrix engine)
├── run_dit_iad_default.sh    # preset: no memory optimizations (> 30 GB VRAM)
└── run_dit_iad_efficient.sh  # preset: all memory optimizations (24 GB VRAM)
```

### Core math (all derived from the official code, `iad/core.py`)

* Linear flow path (`transport/path.py::ICPlan`):
  `z_t = (1−t)·z_data + t·ε`, velocity target `v* = ε − z_data`.
* Clean-latent conversion (velocity parameterization):
  `ẑ = z_t − t·v` — exact at `v = v*`, and exactly one Euler step of the
  probability-flow ODE from `t` to `0`.
* SimpleNet corruption: `z_in = z₀ + m·σ·ε`, conditioning time `t_c = σ/(1+σ)`,
  `ẑ = z_in − t_c·v` (exact at `v* = (z_in − z₀)/t_c`). The `hybrid` loss
  (`L1 + λ·mean_flat((v − v*)²)` on one shared flow draw) still exists in the code
  but no config uses it any more.
* Experiment A does not use these paths for training: it calls the untouched
  official `Transport.training_losses` (unit test F proves the two are the
  same objective when the draw is shared).

## 3. Usage

Environment: cluster conda env `rae` (py3.10, torch 2.8.0); RAE weights under
`rae_fidelity/RAE/models/` (`hf download nyu-visionx/RAE-collections
--local-dir models`); MVTec at `$MVTEC_ROOT`.

```bash
# full matrix (pilot categories: carpet grid hazelnut screw transistor)
bash run_dit_iad_default.sh      # no memory optimizations (needs > 30 GB VRAM)
bash run_dit_iad_efficient.sh    # all memory optimizations (24 GB VRAM)

# both forward to run_dit_iad.sh (shared engine); every knob below still applies

# subset / smoke test
EXPT="A B" CATS="carpet" SEEDS="0" STEPS=500 bash run_dit_iad_efficient.sh

# adaptor ablation only (F vs B, G vs C), and a different adaptor family
EXPT="B F" bash run_dit_iad_efficient.sh
EXPT="G" ADAPTOR=mlp bash run_dit_iad_efficient.sh
# edit `adaptor:` in configs/F_*.yaml / G_*.yaml for other variants
# (type: affine + init: train_stats = PatchCore/FeaNorm-style statistics)

# 24 GB GPU, or force a preset explicitly
MEM=low bash run_dit_iad.sh      # activation recompute + CPU EMA
MEM=full bash run_dit_iad.sh     # no recompute, EMA on GPU (needs > 30 GB)

# large GPU: fastest settings (raise the micro-batch too if memory allows)
MEM=full BATCH=4 GRAD_ACCUM=2 bash run_dit_iad.sh

# eval-only, noise sweep over modes + levels (flow path: A only)
STAGE=eval EXPT="A" MODES="single,multi,ode" T_VALUES="0.1,0.3,0.5,0.7,0.9" \
  REPEATS=4 bash run_dit_iad.sh

# SimpleNet noise level for the SimpleNet arms, extra scoring without retraining
SIGMA=0.05 EXPT="B J" bash run_dit_iad.sh
STAGE=eval EXPT="A B J" SCORES="latent_l2" bash run_dit_iad.sh
```

Direct CLI:

```bash
python rae_dit_iad.py train --config configs/B_image_l1.yaml \
    --mvtec-root $MVTEC_ROOT --out results_dit_iad --categories carpet screw --seed 0

# experiment F with the adaptor swapped from the CLI (SimpleNet by default)
python rae_dit_iad.py train --config configs/F_image_l1_adaptor.yaml \
    --adaptor conv1x1 --adaptor-rank 8 \
    --mvtec-root $MVTEC_ROOT --out results_dit_iad --categories carpet screw --seed 0

python rae_dit_iad.py eval --config configs/B_image_l1.yaml \
    --mvtec-root $MVTEC_ROOT --out results_dit_iad --categories carpet screw \
    --mode single --t 0.5 --seed 0
```

Tests (A–I from the spec, plus J memory switches, K adaptor, L SimpleNet-style
corruption, M decoder-free loss and latent scoring, N discriminator, O anchored adaptor;
weight-independent, CPU):

```bash
pytest rae_dit_iad/tests/test_iad.py -v
```

`eval` reads the checkpoint's embedded config (provenance), so `--config` is
optional after training; checkpoints are EMA weights by default (the raw model
weights are also stored, `--ckpt-which` picks the checkpoint file, EMA usage
follows `training.use_ema`). For F/G the adaptor is rebuilt from that same
embedded config and its EMA weights are loaded, so training and inference always
agree on the latent space.

### GPU memory: two presets (same math, different hardware)

Measured on the cluster 4090: DiT-XL (~840 M params) + released decoder + EMA in
fp32 peaks at ≈22 GB *before* the first Adam state allocation, so even
micro-batch 2/accum 4 OOMs on 24 GB. The fixed cost dominates:

| item | fp32 size |
|---|---|
| DiT params + grads + Adam (m, v) | ≈ 4 × 3.36 GB |
| EMA copy of the DiT | 3.36 GB |
| frozen decoder (+ grads/Adam when fine-tuning) | 1.2 GB (+3.6 GB) |
| fp32 activations (2 × 256 tokens × 30 blocks) | ≈ 6 GB |

So memory is controlled by switches that **do not change the objective**, exposed
as two launch presets:

```bash
bash run_dit_iad_default.sh      # MEM=full: no recompute, EMA on GPU (> 30 GB)
bash run_dit_iad_efficient.sh    # MEM=low: recompute + EMA on CPU (24 GB)
MEM=full BATCH=4 GRAD_ACCUM=2 bash run_dit_iad.sh   # explicit, larger GPU
```

* `--grad-checkpoint / --no-grad-checkpoint` (`training.grad_checkpoint`,
  default `false`) — wraps every DiT block (and the decoder when it is being
  fine-tuned) in `torch.utils.checkpoint`, i.e. activations are **recomputed**
  in backward instead of stored. Identical forward values, ~30 % slower step,
  activation memory drops from ≈6 GB to ≈0.3 GB.
* `--ema-device {cuda,cpu}` (`training.ema_device`, default `cuda`) — `cpu`
  keeps the EMA copy off the GPU (saves 3.36 GB, identical decay math, only
  ~0.1 s/step of PCIe traffic; validation temporarily moves it back).
* `--batch-size` / `--grad-accum` (`BATCH` / `GRAD_ACCUM`) — micro-batch ×
  accumulation; configs default to 2 × 4 = effective batch 8, so optimizer,
  EMA and scheduler still see one update per 8 images.
* Autocast (`training.precision: bf16`) is available but **off**: it is a real
  numerical change and `MEM=low` already fits fp32.

With `MEM=low` the expected peaks are ≈15 GB (A), ≈16 GB (B/D), ≈19 GB (C/E);
`MEM=full` is the fastest setting for a 40–80 GB card (raise `BATCH` too, e.g.
`MEM=full BATCH=4 GRAD_ACCUM=2`). The frozen encoder is always moved to CPU
once its latents are cached (exact: it is never used again), and the runner sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to limit fragmentation.
Eval runs under `no_grad` and uses `EVAL_BATCH` (default 8).


## 4. Evaluation

* **Pixel**: AUROC, AP, AUPRO (`latent_draem.evaluate_maps`, project
  convention: Gaussian blur σ = 4 on 256 px maps).
* **Image**: AUROC and AP with all three aggregations — `mean`, `max`,
  `top1` (mean of the top 1% pixel scores).
* **Inference modes** (cost reported per image in `results.csv`):
  `single` (1 DiT pass, exactly one Euler step), `multi` (`--repeats`
  independent corruptions, maps averaged), `ode` (`--ode-steps` explicit Euler
  steps on the official warped time grid). `multi`/`ode` apply to A only: the
  SimpleNet arms add no test-time noise, so they run `single`.
* **Scores** (`--score`): `pixel` = `mean_c |x − D(ẑ)|`, `latent_cos` /
  `latent_l2` = token distance between `E(x)` and `ẑ` before the decoder, `disc` =
  DRAEM's learned segmentation map (SimpleNet arms; skipped for A).
* **Reproducibility**: `--seed` fixes training init/data order and inference
  noise. Inference noise is per-image-index (`seed·1_000_003 + idx·97 +
  rep·13`), so results are **batch-size invariant** and comparable across model
  variants (unit test H).
* **Adaptor (F/G)** is applied to the clean latents inside `compute_losses`
  (`z0 ← A(z0)`) and to `rae.encode(x)` in `reconstruction`, so both use the
  identical adapted space; `A`'s statistics (if any) are fit on the MVTec
  **train split only** — no test leakage.
* **Outputs** under `results_dit_iad/seed<S>/<exp_id>/`: `joint_{best,last}.pt`,
  `metrics.jsonl`, `results.csv` (deduplicated rows per setting),
  `per_image.csv`, `report.md` (aggregate + per-category tables),
  `viz/<cat>_<exp>_<mode>_t<t>[_<score>].png` grids.

## 5. Deviations from the official RAE repo (all documented in code)

1. **Data/objective**: trained on MVTec normal images instead of ImageNet; the
   encoder is frozen and latents are pre-cached (exact given a frozen encoder).
2. **Null label** (1000) instead of ImageNet class labels — official CFG
   unconditional token; matches official label-dropout behaviour.
3. **Single process, step-based budget** — no DDP, `training.steps` instead of
   epochs; official optimizer/scheduler/EMA hyperparameters otherwise kept.
4. **Config plumbing**: our experiment YAMLs point at the official *training*
   config `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml` (it
   carries the original transport `logit-normal_0_1` + optimizer settings) and
   inject the released checkpoint path via `rae_config_overrides.stage_2.ckpt`
   (the training YAML has no `ckpt` key).
5. **Two-group optimizer** when `decoder_finetune` (decoder lr = 0.1 × base);
   the official `build_scheduler` resets all group lrs, so group lrs are
   re-applied afterwards (`iad.models.fix_param_group_lrs`).
6. **Additive corruption** `z_t = z_0 + σ·ε` (all tokens) is still in the code but
   no config uses it any more; the conditioning time is `σ/(1+σ)`.
7. `ode` inference integrates on the official warped time grid
   `warp(u) = s·u/(1+(s−1)u)`, `s = √(prod(latent)/4096) = √48`.
8. **Feature adaptor (F/G only)** — an addition, not part of the official RAE:
   a small module between the frozen encoder and the DiT/decoder
   (`iad/adaptor.py`). Experiments A–E are untouched by construction: with no
   `adaptor:` block (or `type: none`) the module is never built, and even with
   one the residual head is zero-initialized, so step 0 of F/G is numerically
   identical to B/C (unit test K asserts this to 1e-12).
9. **Memory-only switches, off by default** (`MEM=full`): activation
   recomputation (`grad_checkpoint`) and a CPU-resident EMA (`ema_device: cpu`)
   change where work/compute happens, not the math — the recomputed forward is
   bit-identical up to deterministic kernels and the EMA decay is the same
   per-parameter update. `MEM=low` enables both to fit a 24 GB card. The frozen
   encoder is always offloaded to CPU after latent caching (it is unused
   afterwards).

10. **SimpleNet-style corruption is the project default** (B–J) — an addition,
   not part of the official RAE: masked additive noise `z₀ + m·σ·ε` instead of the
   flow path, conditioning time σ/(1+σ), no noise at test time. A alone keeps the
   official flow path and loss, untouched.
11. **Latent loss and latent scoring** — additions: `loss: latent` (J) and the
   `lambda_latent` term (D/E) train on a token MSE to the clean latent;
   `score: latent_*` scores before the decoder. Neither touches the official
   objective or decoder.
12. **Discriminator (all SimpleNet arms)** — an addition: DRAEM's discriminative
   sub-network on the token grid, trained with a focal loss against the synthetic
   mask; detached by default, so it never changes the DiT's training.

## 6. Known limitations

* **Experiment A ignores the config `noise:` block during training** — the
  official `training_losses` draws its own `(t, ε)` from the transport config
  (`logit-normal_0_1` + official warp). A is the only flow-path arm; every other
  arm reads the SimpleNet keys of its `noise:` block.
* Image L1 at the *validation* step selects `best` checkpoints by validation
  loss (`val_img` / `val_fm`), not by anomaly-detection AUROC — no test-set
  leakage, but the best-loss checkpoint is not necessarily the best detector.
* Decoder fine-tuning (C/E) has no proximal/anchor regularizer to the released
  decoder weights; with few MVTec images it can overfit (compare against B).
* EMA covers the DiT (and the adaptor in F/G); a finetuned decoder is
  checkpointed raw.
* **Adaptor variants are not apples-to-apples in one respect**: with a *frozen*
  decoder (F) the adaptor must stay near the decoder's input manifold, so a
  `train_stats` affine (PatchCore/FeaNorm-style whitening) or a non-residual
  re-mapping is only meaningful with a fine-tuned decoder (G). F and G use the
  identical SimpleNet adaptor by design.
* The adaptor is tiny (~0.3 M params at `rank: 4`) next to the 840 M DiT, so a
  *null* result (F ≈ B) is the expected outcome unless the domain gap is the
  bottleneck; the BatchNorm in `simplenet` uses batch statistics during
  training and running statistics at validation/inference (SimpleNet's own
  convention).
* Per-category runs (`--scope per_category`) reset the DiT to the official
  initialization and the decoder to its released weights for every category.
* Joint training mixes all categories (categories are not model classes — the
  DiT still sees only the null label).
* Pixel maps are the channel-averaged `|x − x̂|` before blurring; no feature
  or perceptual metrics.
* Runs are CPU/GPU single-device; `fp16`/`bf16` are opt-in
  (`training.precision`), default `fp32`.

## 7. Status

**First full run (job 152776, 5 pilot categories, seed 0, 4000 steps).** Means over
carpet, grid, hazelnut, screw, transistor (image AUROC max / pixel AUROC / AUPRO):

| Arm · score | Image AUROC | Pixel AUROC | AUPRO |
|---|---|---|---|
| kNN, frozen DINOv2-B tokens | 0.969 | 0.956 | 0.861 |
| latent_draem `latent_noise` restorer, residual | 0.972 | 0.965 | 0.842 |
| **A · latent_cos** | **0.964** | **0.963** | **0.879** |
| J · latent_cos | 0.764 | 0.758 | 0.339 |
| best pixel score (E) | 0.698 | 0.812 | 0.610 |
| disc, every arm | 0.48–0.60 | ≈ 0.71 | ≤ 0.37 |

- Only A works (ties kNN, slightly better AUPRO). Every SimpleNet arm at σ = 0.1 fails:
  the corruption is near-identity, so the restorer copies real defects through, and the
  DRAEM head learns to find faint jitter that never occurs at test time. latent_draem's
  working range was σ ∈ [0.5, 1.5] → **J2 / K2**.
- Decoded (pixel) scores are weak in every arm, A included (0.62 vs 0.96 with latent_cos).
- latent_cos is inverted (< 0.5) for the image-L1-only arms B, C, F, G: without a latent
  term ẑ drifts freely where the decoder does not care.
- That run took ~11.5 h per arm because `validate()` moved the GPU EMA to the CPU after
  step 250 (`cuda:0` != `cuda`); fixed in `train.modules_to_move` (Test P).

**Job 153649 (`PLAN=next`).** A is stable over seeds 0–2: image AUROC 0.963 ± 0.001,
pixel AUROC 0.962 ± 0.001, AUPRO 0.878 ± 0.002 (kNN 0.861). Strong noise helped J and K
but not enough: J2 latent_cos 0.815 (AUPRO 0.654), K2 0.835 (0.665); the DRAEM head now
learns (J2 disc 0.814, AUPRO 0.724; K2 0.818, 0.762). Both fail where defects are
plausible features in the wrong place (screw ≈ 0.58, transistor ≈ 0.72).

**Scope.** Jobs 152776 and 153649 trained **one joint model** for the five categories --
the multi-class (unified) setting, a different problem from the standard one-class
benchmark. Every config is now `scope: per_category` (one DiT per category, reset to
the released weights each time), as are both references (kNN memory bank and
latent_draem). The joint results stay under `results_dit_iad/seed*/`; per-category
runs write to `results_dit_iad/per_category/seed*/`. `SCOPE=joint` restores the old
behaviour.

**Heavy files.** Checkpoints go to `CKPT_ROOT` (default
`/po1/rakhimov/training/rae_dit_iad/`, same `[per_category/]seed<S>/<exp_id>/` layout);
only CSVs, reports, logs and grids stay in the repo. Per-category runs save one
eval-only checkpoint per category (`<cat>_best.pt`: EMA weights, no optimizer state,
~3.5 GB, not resumable) and skip the periodic 13 GB resume points, whose writes stalled
training for minutes each. With `KEEP_CKPT=0` (default) each category is trained,
evaluated with every score and then deleted before the next one starts, so a run holds
at most one checkpoint at a time; `KEEP_CKPT=1` keeps them (e.g. for later re-scoring).

**Next (`PLAN=all`, the SLURM default):** every experiment so far, per category --
A (seeds 0–2), B–G, J, K, the three noise ablations, J2, K2, J3 token swap, J4 DTD
texture paste, J5 mix, K5 mix + adaptor (20 runs × 5 category models). DTD is read
from `DTD_ROOT` (default `../latent_draem/data/dtd`, downloaded once if empty).
