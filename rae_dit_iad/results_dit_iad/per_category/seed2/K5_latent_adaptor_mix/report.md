# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.5251 | 0.8646 | 0.8569 | 0.8094 | 0.9538 | 0.9509 | 0.9315 | 0.4410 | 0.8028 |
| single | 1.0 | 1 | best | latent_cos | 0.8792 | 0.9612 | 0.9599 | 0.9500 | 0.9864 | 0.9858 | 0.9821 | 0.3351 | 0.9118 |
| single | 1.0 | 1 | best | pixel | 0.4379 | 0.6231 | 0.6179 | 0.7337 | 0.8461 | 0.8391 | 0.7683 | 0.0588 | 0.4332 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6256 | 0.6091 | 0.6589 | 0.8694 | 0.8468 | 0.8751 | 0.6754 | 0.0964 | 0.3181 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9824 | 0.3950 | 0.9266 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2953 | 0.9346 | 0.9169 | 0.7275 | 0.9803 | 0.9755 | 0.9882 | 0.7512 | 0.9476 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2503 | 0.6370 | 0.5770 | 0.5980 | 0.8454 | 0.8030 | 0.8611 | 0.0213 | 0.5482 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7584 | 0.9223 | 0.9199 | 0.9000 | 0.9729 | 0.9716 | 0.9819 | 0.2752 | 0.8970 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.7549 | 0.7946 | 0.7969 | 0.8913 | 0.9272 | 0.9264 | 0.8748 | 0.1309 | 0.6579 |
