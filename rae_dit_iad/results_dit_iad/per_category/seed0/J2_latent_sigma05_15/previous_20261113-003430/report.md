# RAE-DiT IAD results (J2_latent_sigma05_15)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.9980 | 0.8945 | 0.9342 | 0.9994 | 0.9598 | 0.9766 | 0.9272 | 0.2095 | 0.8638 |
| single | 1.0 | 1 | best | latent_cos | 0.9952 | 0.9948 | 0.9952 | 0.9985 | 0.9983 | 0.9984 | 0.9612 | 0.3284 | 0.8084 |
| single | 1.0 | 1 | best | pixel | 0.4996 | 0.5084 | 0.5453 | 0.8150 | 0.8041 | 0.8340 | 0.3473 | 0.0114 | 0.0675 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.4996 | 0.5084 | 0.5453 | 0.8150 | 0.8041 | 0.8340 | 0.3473 | 0.0114 | 0.0675 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9952 | 0.9948 | 0.9952 | 0.9985 | 0.9983 | 0.9984 | 0.9612 | 0.3284 | 0.8084 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9980 | 0.8945 | 0.9342 | 0.9994 | 0.9598 | 0.9766 | 0.9272 | 0.2095 | 0.8638 |
