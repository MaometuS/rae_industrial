# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.3897 | 0.8415 | 0.8248 | 0.7414 | 0.9452 | 0.9405 | 0.9357 | 0.4559 | 0.8027 |
| single | 1.0 | 1 | best | latent_cos | 0.8707 | 0.9524 | 0.9522 | 0.9467 | 0.9842 | 0.9837 | 0.9783 | 0.3259 | 0.9013 |
| single | 1.0 | 1 | best | pixel | 0.4498 | 0.6419 | 0.6280 | 0.7388 | 0.8552 | 0.8453 | 0.7701 | 0.0617 | 0.4434 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6392 | 0.6437 | 0.6754 | 0.8761 | 0.8631 | 0.8821 | 0.6779 | 0.1018 | 0.3350 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9804 | 0.3868 | 0.9229 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2203 | 0.9077 | 0.8820 | 0.7059 | 0.9734 | 0.9670 | 0.9853 | 0.7060 | 0.9361 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2603 | 0.6401 | 0.5807 | 0.6014 | 0.8473 | 0.8085 | 0.8622 | 0.0216 | 0.5518 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7413 | 0.9049 | 0.9045 | 0.8934 | 0.9683 | 0.9674 | 0.9762 | 0.2650 | 0.8797 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.5591 | 0.7754 | 0.7676 | 0.7769 | 0.9171 | 0.9140 | 0.8861 | 0.2057 | 0.6692 |
