# RAE-DiT IAD results (J2_latent_sigma05_15)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.8850 | 0.8915 | 0.9007 | 0.9599 | 0.9638 | 0.9674 | 0.9164 | 0.2263 | 0.7915 |
| single | 1.0 | 1 | best | latent_cos | 0.8360 | 0.8838 | 0.8937 | 0.9264 | 0.9590 | 0.9637 | 0.9679 | 0.3023 | 0.8728 |
| single | 1.0 | 1 | best | pixel | 0.4035 | 0.5810 | 0.5778 | 0.7206 | 0.8281 | 0.8308 | 0.6027 | 0.0116 | 0.3014 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.5108 | 0.5205 | 0.5534 | 0.8247 | 0.8117 | 0.8393 | 0.3510 | 0.0115 | 0.0718 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9952 | 0.9980 | 0.9976 | 0.9985 | 0.9994 | 0.9993 | 0.9718 | 0.3716 | 0.8960 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9984 | 0.9775 | 0.9900 | 0.9995 | 0.9928 | 0.9969 | 0.9527 | 0.2709 | 0.8812 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2962 | 0.6415 | 0.6022 | 0.6166 | 0.8444 | 0.8222 | 0.8544 | 0.0117 | 0.5309 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.6768 | 0.7696 | 0.7897 | 0.8543 | 0.9186 | 0.9282 | 0.9640 | 0.2330 | 0.8496 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.7717 | 0.8055 | 0.8114 | 0.9202 | 0.9347 | 0.9379 | 0.8801 | 0.1817 | 0.7019 |
