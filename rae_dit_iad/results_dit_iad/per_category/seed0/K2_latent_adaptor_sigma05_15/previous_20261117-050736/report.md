# RAE-DiT IAD results (K2_latent_adaptor_sigma05_15)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.9819 | 0.8302 | 0.8656 | 0.9911 | 0.9294 | 0.9472 | 0.8984 | 0.1481 | 0.8072 |
| single | 1.0 | 1 | best | latent_cos | 0.9950 | 0.9955 | 0.9986 | 0.9985 | 0.9985 | 0.9995 | 0.9554 | 0.2005 | 0.8285 |
| single | 1.0 | 1 | best | pixel | 0.4605 | 0.4713 | 0.4812 | 0.7606 | 0.7702 | 0.7679 | 0.4538 | 0.0098 | 0.1501 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.5201 | 0.5157 | 0.5638 | 0.8297 | 0.8102 | 0.8455 | 0.3674 | 0.0121 | 0.0788 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9900 | 0.9944 | 0.9980 | 0.9970 | 0.9982 | 0.9994 | 0.9504 | 0.2496 | 0.7952 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9739 | 0.6922 | 0.7512 | 0.9861 | 0.8709 | 0.9018 | 0.8641 | 0.1270 | 0.7738 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4010 | 0.4269 | 0.3985 | 0.6915 | 0.7302 | 0.6903 | 0.5402 | 0.0076 | 0.2214 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 0.9967 | 0.9992 | 1.0000 | 0.9988 | 0.9997 | 0.9605 | 0.1514 | 0.8618 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.9900 | 0.9683 | 0.9799 | 0.9962 | 0.9880 | 0.9925 | 0.9326 | 0.1692 | 0.8406 |
