# RAE reconstruction-fidelity pilot (MVTec AD)

Checks whether a **frozen RAE** (frozen DINOv2-B / SigLIP-B / MAE-B encoder + released ViT decoder) can reproduce
normal industrial images well enough to serve as the backbone of a generate-anomaly → remove-anomaly pipeline.
An SD-VAE (`stabilityai/sd-vae-ft-ema`) is included as the reference latent.

## Run (GPU node)

```bash
MVTEC_ROOT=/path/to/mvtec_ad bash run_cluster.sh
```

The script clones the official repo (`bytetriper/RAE`), installs it, downloads `nyu-visionx/RAE-collections`,
checks one config with the repo's own `stage1_sample.py`, then runs every config in `configs/stage1/pretrained/`.
Models are loaded exactly as the official script does (`parse_configs` → `instantiate_from_config` → `encode`/`decode`,
input in [0,1]). Runtime is minutes per model: about 1,700 test images + 50 train/good images per category at 256 px.

Manual run / subset:

```bash
python rae_fidelity.py --rae-root RAE --configs configs/stage1/pretrained/<name>.yaml \
    --mvtec-root $MVTEC_ROOT --categories screw carpet --out results_quick
python rae_fidelity.py --mock --mvtec-root $MVTEC_ROOT --out results_mock   # pipeline test, no weights
```

## Outputs

- `report.md`: mean table plus per-category pivots for PSNR, SSIM and pass-through
- `per_category.csv`, `per_image.csv`: all numbers
- `<model>/grids/<cat>_{normal,defect}.png`: input | reconstruction | error heat map (shared scale = 3× normal p99) | GT mask

## What to look at

| Metric | Question it answers | What would be bad |
|---|---|---|
| `normal_psnr`, `normal_ssim` | Does the frozen decoder keep industrial detail? | Clearly below SD-VAE, especially on textures (carpet, grid, leather, wood) |
| `normal_err_p99` | Noise floor that real defects must exceed later | Comparable to the error in subtle defect regions |
| `anom_psnr_in_mask`, `passthrough_ratio` | Does the decoder reproduce defects? (≈1 = yes) | Not bad in itself: the restorer must learn removal. It matters only if defects get "removed" by blur, which would mean the latent can't represent them at all |
| `pixel/image_auroc_raw` | Raw error as a detector | Sanity check only |

Notes: 256 px is the decoders' native resolution. Thin MVTec defects (screw, capsule) may disappear at this size
for every model, so read the grids before trusting the averages. To check 512 px, add a 512 decoder config if one is
released and pass `--size 512`.
