# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.2836 | 0.8562 | 0.8485 | 0.7179 | 0.9540 | 0.9518 | 0.9876 | 0.5018 | 0.9414 |
| single | 1.0 | 1 | best | latent_cos | 0.9980 | 0.9996 | 0.9996 | 0.9994 | 0.9999 | 0.9999 | 0.9864 | 0.3663 | 0.9515 |
| single | 1.0 | 1 | best | pixel | 0.6532 | 0.7073 | 0.7064 | 0.8262 | 0.8840 | 0.8715 | 0.7150 | 0.1032 | 0.3976 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.8226 | 0.7905 | 0.8772 | 0.9451 | 0.9211 | 0.9589 | 0.6693 | 0.0934 | 0.3092 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9960 | 0.9992 | 0.9992 | 0.9989 | 0.9998 | 0.9998 | 0.9852 | 0.4217 | 0.9489 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2179 | 0.7416 | 0.7295 | 0.6919 | 0.9185 | 0.9152 | 0.9910 | 0.7582 | 0.9484 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4837 | 0.6241 | 0.5355 | 0.7072 | 0.8470 | 0.7840 | 0.7606 | 0.1130 | 0.4860 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9875 | 0.3110 | 0.9541 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.3492 | 0.9708 | 0.9674 | 0.7439 | 0.9895 | 0.9883 | 0.9843 | 0.2454 | 0.9344 |
