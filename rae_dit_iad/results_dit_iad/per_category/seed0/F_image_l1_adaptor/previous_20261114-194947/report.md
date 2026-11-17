# RAE-DiT IAD results (F_image_l1_adaptor)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.7713 | 0.8098 | 0.7976 | 0.9166 | 0.9350 | 0.9305 | 0.7326 | 0.0467 | 0.4164 |
| single | 0.1 | 1 | best | latent_cos | 0.1223 | 0.3293 | 0.3031 | 0.5723 | 0.6802 | 0.6738 | 0.0226 | 0.0058 | 0.0004 |
| single | 0.1 | 1 | best | pixel | 0.5504 | 0.6453 | 0.6727 | 0.7873 | 0.8465 | 0.8585 | 0.6536 | 0.1234 | 0.4952 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.3447 | 0.4535 | 0.4290 | 0.6995 | 0.7736 | 0.7518 | 0.3631 | 0.0162 | 0.1161 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0959 | 0.3202 | 0.3046 | 0.5829 | 0.6927 | 0.6901 | 0.0294 | 0.0081 | 0.0007 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8768 | 0.7632 | 0.7548 | 0.9634 | 0.9261 | 0.9226 | 0.7321 | 0.0443 | 0.3992 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.7561 | 0.8371 | 0.9165 | 0.8750 | 0.9194 | 0.9653 | 0.9441 | 0.2307 | 0.8743 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.1487 | 0.3383 | 0.3016 | 0.5616 | 0.6677 | 0.6575 | 0.0157 | 0.0035 | 0.0000 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.6658 | 0.8563 | 0.8404 | 0.8697 | 0.9438 | 0.9384 | 0.7330 | 0.0491 | 0.4335 |
