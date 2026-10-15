# RAE as a reconstruction-based detector: anti-copying ablation

```
x ─► frozen DINOv2-B encoder ─► z ─► [S] [M] [N] ─► RAE ViT-XL decoder (fine-tuned) ─► x̂ ≈ normal image
                                                                         score: compare x with x̂
```

**Why mechanisms are needed:** the RAE latent has 16 × 16 × 768 = 196,608 values, the same as a 256×256 RGB image.
There is no bottleneck, so a decoder fine-tuned on normal images can copy defects (Exp 1: every decoder reproduced
the hazelnut crack).

| Mechanism | Training | Test |
|---|---|---|
| **M** masking | 25–50% of tokens replaced by a learned mask token | 4 passes; tokens split by grid parity; each 16×16 patch is taken from the pass where its token was masked |
| **N** noisy bottleneck | Feature dropout (p = 0.2) + Gaussian noise σ ~ U(0, 0.8) on tokens | No noise |
| **S** synthetic anomalies | Latent/pixel anomalies (`mix` from latent_draem); target is the clean image | No synthesis |
| **none** | Plain decoder fine-tuning on normal images (naive baseline) | Plain reconstruction |

All 8 combinations run (`none M N S MN MS NS MNS`). The encoder stays frozen, so RAE's latent space and DiT stay valid.
The decoder restarts from the released RAE weights for every run. Loss: L1 + 0.5·(1 − SSIM). 2000 steps, batch 8,
learning rate 2e-5 (1e-3 for the mask token).

**Scores**
- `pix`: |x − x̂|, blurred
- `ssim`: 1 − SSIM(x, x̂)
- `feat`: 1 − cos(E(x), E(x̂)) per token, i.e. re-encode the reconstruction
- `knn`: frozen-token control, independent of the combination
- Fusions, z-normalized on 20 held-out normal images: `pix+feat`, `feat+knn`, `pix+feat+knn`

## Run

```bash
CATS="screw" COMBOS="none MNS" STEPS=200 OUT=results_recon_smoke bash run_recon_ablation.sh   # smoke test first
bash run_recon_ablation.sh                                                                    # 5 cats × 8 combos × 1 seed
SEEDS="0 1 2" COMBOS="none <best>" bash run_recon_ablation.sh                                  # add seeds for the best ones
```

- It reuses the `rae` env, the RAE weights in `../rae_fidelity/RAE`, the DTD download in `../latent_draem/data/dtd`, and the
  shared utilities in `../latent_draem/latent_draem.py`. That file must be the current version; the script checks.
- Runtime is not measured yet. Fine-tuning a ViT-XL decoder for 2000 steps is likely 10–20 min per run, so roughly 7–13 h
  for 40 runs. The smoke test prints the real time per run.
- Results are written after every run, and finished runs are skipped on restart.

## Outputs (`results_recon_ablation/`)
- `report.md`: every combination × score (mean over categories), per-category AUPRO tables, and the gain of each
  combination over `none`
- `results.csv`: every number
- `viz/<cat>_<combo>.png`: input | x̂ | pix map | feat map | ground truth (4 defective + 2 normal images)

## What to look for
1. **`none` vs the rest:** how much each mechanism adds over plain fine-tuning. This is the "why not just fine-tune RAE?" answer.
2. **Single mechanisms vs combinations:** whether they stack or overlap.
3. **`feat` vs `pix`:** whether re-encoding tolerates the decoder's remaining pixel errors better.
4. **Best combination vs `knn`,** and the fusions: whether the RAE reconstruction adds signal beyond the frozen features.
5. **x̂ in the grids:** do defects disappear? Removal is now visible directly in pixels.
