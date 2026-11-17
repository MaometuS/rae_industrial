# RAE-DiT IAD results (K_latent_simplenet_adaptor)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.8784 | 0.8788 | 0.8768 | 0.9621 | 0.9653 | 0.9651 | 0.7713 | 0.0763 | 0.5409 |
| single | 0.1 | 1 | best | latent_cos | 0.8816 | 0.7693 | 0.7837 | 0.9646 | 0.9271 | 0.9313 | 0.8695 | 0.0843 | 0.5237 |
| single | 0.1 | 1 | best | pixel | 0.5056 | 0.5076 | 0.5522 | 0.8201 | 0.8056 | 0.8368 | 0.3413 | 0.0114 | 0.0686 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5056 | 0.5076 | 0.5522 | 0.8201 | 0.8056 | 0.8368 | 0.3413 | 0.0114 | 0.0686 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.8816 | 0.7693 | 0.7837 | 0.9646 | 0.9271 | 0.9313 | 0.8695 | 0.0843 | 0.5237 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8784 | 0.8788 | 0.8768 | 0.9621 | 0.9653 | 0.9651 | 0.7713 | 0.0763 | 0.5409 |
