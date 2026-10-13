# Latent DRAEM: synthesize and remove anomalies in a frozen RAE latent

DRAEM works in pixels and DSR in a VQ latent it trains itself. This prototype does both steps in the **frozen RAE token
space** (16×16×768 for DINOv2-B and for MAE). The pixel decoder isn't used for scoring, so the fidelity problem from the
first pilot doesn't affect detection.

```
x ──E (frozen RAE encoder)──► z ──synth──► z_a ──R (token transformer)──► r ≈ z
                                             └──────────┬──────────────────┘
                                                        D([z_a, r, z_a−r, cos]) ──► token anomaly logits ──► upsample + blur
```

| Mode | What it does (no real defects are used) |
|---|---|
| `pixel` | DRAEM in pixel space (DTD texture in a Perlin mask), then encoded. The pixel-synthesis reference |
| `latent_tex` | DTD texture **tokens** blended into a Perlin token mask: `z + β·m·(z_tex − z)` |
| `latent_self` | Tokens from another normal image of the same category, rolled/flipped. Latent CutPaste for misplaced or structural content |
| `latent_noise` | Gaussian noise on the masked tokens (σ in normalized latent units). Latent variance inflation |
| `mix` | One of the above, chosen per sample |

Scores reported for every run:
- `disc`: the learned discriminator map (DRAEM-style). This is the main score.
- `resid`: `1 − cos(z, R(z))`, the restorer residual with no discriminator.
- `knn`: **the control.** Nearest-neighbour distance from each test token to the bank of training tokens (PatchCore-lite).
  It needs no training and tells you how far the frozen latent gets on its own. If `disc` doesn't beat `knn`, the
  restorer and synthesis add nothing.

Metrics: image AUROC (max of the map), pixel AUROC, pixel AP, AUPRO@30% FPR, all at 256 px.

## Run

```bash
bash run_latent_draem.sh                                                # full pilot: 2 encoders × 5 categories × 5 modes
CATS="screw" MODES="pixel latent_tex" STEPS=500 bash run_latent_draem.sh   # smoke test first
```

It reuses the `rae` conda env and the RAE repo and weights from `../rae_fidelity/RAE` (override with `RAE_ROOT`). DTD
downloads into `data/dtd` the first time it runs. Each (encoder, category, mode) run trains for 3000 steps, roughly
1–3 min on one GPU, so the full pilot should take about 1–2 h. Results are written after every run and finished runs are
skipped, so an interrupted job can simply be restarted.

## Outputs (`results_latent_draem/`)
- `report.md`: mean over categories for each encoder (mode × score), plus per-category tables for `disc` and `knn`
- `results.csv`: every number
- `<encoder>/viz/<cat>_<mode>.png`: input | decode(z) | **decode(R(z))** (what the restorer thinks the normal image is) | disc map | GT

## What to look for
1. **`disc` vs `knn`** on each encoder. This answers whether the method is worth pursuing.
2. **latent modes vs `pixel`.** If `latent_tex`/`latent_self`/`mix` match or beat `pixel`, synthesizing in the latent space
   is viable, and a learned generator (the second RAE/DiT) can later replace these hand-made perturbations.
3. **DINOv2-B vs MAE.** This shows whether a more semantic latent helps when pixel fidelity no longer matters.
4. **AUPRO and AP on screw**, the small-defect category. This is where the 16×16 token grid will hurt.

## Known limitations of v1
- Each category is trained separately, and anomalies can land on the background (no foreground mask).
- The token grid is 16×16 (224 px into DINOv2, 256 px into MAE). Thin defects will be blurred.
- The discriminator uses soft per-token targets (the fraction of each token covered by the mask). `--hard-targets` switches.
