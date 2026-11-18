# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.2352 | 0.9073 | 0.8852 | 0.7140 | 0.9728 | 0.9671 | 0.9864 | 0.7187 | 0.9397 |
| single | 1.0 | 1 | best | latent_cos | 1.0000 | 0.9996 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 0.9836 | 0.3969 | 0.9419 |
| single | 1.0 | 1 | best | pixel | 0.6818 | 0.6810 | 0.7348 | 0.8915 | 0.8773 | 0.9069 | 0.6858 | 0.1049 | 0.3320 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6818 | 0.6810 | 0.7348 | 0.8915 | 0.8773 | 0.9069 | 0.6858 | 0.1049 | 0.3320 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 0.9996 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 0.9836 | 0.3969 | 0.9419 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2352 | 0.9073 | 0.8852 | 0.7140 | 0.9728 | 0.9671 | 0.9864 | 0.7187 | 0.9397 |
