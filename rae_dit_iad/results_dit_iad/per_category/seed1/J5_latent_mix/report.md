# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.4134 | 0.8304 | 0.8142 | 0.7585 | 0.9416 | 0.9357 | 0.9283 | 0.4537 | 0.7870 |
| single | 1.0 | 1 | best | latent_cos | 0.8868 | 0.9577 | 0.9565 | 0.9536 | 0.9859 | 0.9853 | 0.9818 | 0.3311 | 0.9170 |
| single | 1.0 | 1 | best | pixel | 0.4636 | 0.6604 | 0.6589 | 0.7441 | 0.8618 | 0.8579 | 0.7755 | 0.0640 | 0.4445 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6818 | 0.6810 | 0.7348 | 0.8915 | 0.8773 | 0.9069 | 0.6858 | 0.1049 | 0.3320 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 0.9996 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 0.9836 | 0.3969 | 0.9419 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2352 | 0.9073 | 0.8852 | 0.7140 | 0.9728 | 0.9671 | 0.9864 | 0.7187 | 0.9397 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2453 | 0.6399 | 0.5831 | 0.5966 | 0.8462 | 0.8090 | 0.8652 | 0.0230 | 0.5569 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7735 | 0.9158 | 0.9131 | 0.9071 | 0.9719 | 0.9705 | 0.9799 | 0.2654 | 0.8921 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.5917 | 0.7534 | 0.7432 | 0.8031 | 0.9104 | 0.9043 | 0.8702 | 0.1887 | 0.6343 |
