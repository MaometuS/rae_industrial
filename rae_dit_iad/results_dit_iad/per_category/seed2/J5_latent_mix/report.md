# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.2813 | 0.9165 | 0.9053 | 0.7290 | 0.9759 | 0.9729 | 0.9848 | 0.7273 | 0.9397 |
| single | 1.0 | 1 | best | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9822 | 0.3877 | 0.9283 |
| single | 1.0 | 1 | best | pixel | 0.6677 | 0.6545 | 0.7159 | 0.8862 | 0.8652 | 0.8981 | 0.6723 | 0.0971 | 0.3093 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6677 | 0.6545 | 0.7159 | 0.8862 | 0.8652 | 0.8981 | 0.6723 | 0.0971 | 0.3093 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9822 | 0.3877 | 0.9283 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2813 | 0.9165 | 0.9053 | 0.7290 | 0.9759 | 0.9729 | 0.9848 | 0.7273 | 0.9397 |
