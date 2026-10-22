# RAE-DiT for Industrial Anomaly Detection (IAD)

Trains the **original RAE DiT** (DiT-DH-XL, `bytetriper/RAE`) on MVTec AD
*normal* images with one of three objectives and evaluates reconstruction-based
anomaly detection. This is the implementation + experiment suite for the
research spec: *image-space L1 reconstruction loss through the RAE decoder
versus the original flow-matching baseline.*

Nothing in the official RAE repo is modified; it is consumed as a library
(`../rae_fidelity/RAE`, commit `a4d18c4`), and MVTec loading / benchmark
metrics come from `../latent_draem/latent_draem.py`.

---

## 1. Experiments

| id | config | loss | decoder | notes |
|----|--------|------|---------|-------|
| **A** | `configs/A_flow_matching.yaml` | official flow matching (untouched `transport.training_losses`) | frozen | baseline, reproduces the original RAE objective on MVTec |
| **B** | `configs/B_image_l1.yaml` | image L1 through the decoder | frozen | gradients flow *through* the frozen decoder into the DiT |
| **C** | `configs/C_image_l1_decfinetune.yaml` | image L1 | **finetuned** | decoder lr = 0.1 × DiT lr (`decoder_lr_ratio`) |
| **D** | `configs/D_hybrid.yaml` | `L1 + λ·L_FM` (λ = 0.1, `--lambda-fm`) | frozen | one shared (t, ε) draw / one DiT forward for both terms |
| **E** | `configs/E_hybrid_decfinetune.yaml` | `L1 + λ·L_FM` | **finetuned** | |
| **F** | `configs/F_image_l1_adaptor.yaml` | image L1 + **feature adaptor** | frozen | SimpleNet-style adaptor after the encoder; compare **F vs B** |
| **G** | `configs/G_image_l1_adaptor_decfinetune.yaml` | image L1 + **feature adaptor** | **finetuned** | identical adaptor to F; compare **G vs C** |

Shared: encoder **always frozen** (and latents pre-cached — exact, since the
encoder never updates), official AdamW hyperparameters (lr 2e-4, betas
(0.9, 0.95), wd 0), official linear warmup/decay scheduler, official EMA
(0.9995), grad clip 1.0, class conditioning via the official CFG **null label**
(1000) because MVTec has no ImageNet label (a no-op w.r.t. the official label
dropout).

Noise ablations (train-time corruption, all based on B):

| config | corruption | time distribution |
|--------|-----------|-------------------|
| `noise_fixed_t03.yaml` | flow | t fixed 0.3 |
| `noise_fixed_t07.yaml` | flow | t fixed 0.7 |
| `noise_range_02_08.yaml` | flow | t ~ U[0.2, 0.8] |
| `noise_additive_sigma03.yaml` | additive `z0 + σ·ε` (documented deviation) | σ fixed 0.3 |

Eval-time noise sweeps use `eval --t 0.1,0.3,0.5,0.7,0.9` (no retraining).

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

## 2. Layout

```
rae_dit_iad/
├── iad/
│   ├── core.py       # corruption, time sampling, velocity→clean conversion, losses
│   ├── models.py     # official RAE/DiT/transport builders, freezing, optimizer
│   ├── adaptor.py    # feature adaptors after the frozen encoder (experiments F/G)
│   ├── train.py      # training engine (EMA, official scheduler, checkpoint/resume)
│   └── infer.py      # single/multi/ode reconstruction, anomaly maps, viz grids
├── rae_dit_iad.py    # CLI: `train` and `eval` subcommands
├── configs/          # experiment + noise-ablation YAMLs
├── tests/test_iad.py # Tests A–I (+J: memory switches, +K: feature adaptor)
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
* Hybrid loss: `L = L1(x̂, x) + λ·mean_flat((v − v*)²)` on a **single shared**
  `(t, ε)` draw, so the FM term is algebraically the official loss.
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

# eval-only, noise sweep over modes + levels
STAGE=eval EXPT="B" MODES="single,multi,ode" T_VALUES="0.1,0.3,0.5,0.7,0.9" \
  REPEATS=4 bash run_dit_iad.sh
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

Tests (Tests A–I from the spec, plus J for the memory switches and K for the
adaptor; weight-independent, CPU):

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
  steps on the official warped time grid).
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
  `viz/<cat>_<exp>_<mode>_t<t>.png` grids.

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
6. **Additive corruption** `z_t = z_0 + σ·ε` exists only for noise ablations;
   the conditioning time becomes `σ/(1+σ)` and `hybrid` loss rejects it (no
   linear-path velocity target exists there).
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

## 6. Known limitations

* **Experiment A ignores the config `noise:` block during training** — the
  official `training_losses` draws its own `(t, ε)` from the transport config
  (`logit-normal_0_1` + official warp). The `noise:` block only drives
  B/C/D/E and validation.
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

Implementation + configs + tests written (A–G + noise ablations + Tests A–K);
cluster execution pending (this subproject is run entirely on the cluster —
`run_dit_iad_default.sh` / `run_dit_iad_efficient.sh`). Result tables land in
`results_dit_iad/seed*/<exp_id>/report.md` after the matrix runs; the adaptor
question is answered by the paired comparisons **F vs B** (frozen decoder) and
**G vs C** (fine-tuned decoder) at the same seed, step budget and inference
settings.
