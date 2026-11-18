# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.2953 | 0.9346 | 0.9169 | 0.7275 | 0.9803 | 0.9755 | 0.9882 | 0.7512 | 0.9476 |
| single | 1.0 | 1 | best | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9824 | 0.3950 | 0.9266 |
| single | 1.0 | 1 | best | pixel | 0.6256 | 0.6091 | 0.6589 | 0.8694 | 0.8468 | 0.8751 | 0.6754 | 0.0964 | 0.3181 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6256 | 0.6091 | 0.6589 | 0.8694 | 0.8468 | 0.8751 | 0.6754 | 0.0964 | 0.3181 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9824 | 0.3950 | 0.9266 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2953 | 0.9346 | 0.9169 | 0.7275 | 0.9803 | 0.9755 | 0.9882 | 0.7512 | 0.9476 |
