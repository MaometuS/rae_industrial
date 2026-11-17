# RAE-DiT IAD results (K2_latent_adaptor_sigma05_15)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.8667 | 0.8979 | 0.9035 | 0.9494 | 0.9656 | 0.9674 | 0.9081 | 0.1865 | 0.7759 |
| single | 1.0 | 1 | best | latent_cos | 0.8535 | 0.9038 | 0.9065 | 0.9330 | 0.9678 | 0.9691 | 0.9716 | 0.3329 | 0.8829 |
| single | 1.0 | 1 | best | pixel | 0.3989 | 0.5806 | 0.5738 | 0.7189 | 0.8267 | 0.8296 | 0.6084 | 0.0118 | 0.3036 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.5072 | 0.5201 | 0.5478 | 0.8239 | 0.8100 | 0.8386 | 0.3624 | 0.0119 | 0.0762 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9972 | 0.9972 | 1.0000 | 0.9992 | 0.9991 | 1.0000 | 0.9767 | 0.4104 | 0.9113 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9968 | 0.9727 | 0.9835 | 0.9990 | 0.9911 | 0.9949 | 0.9297 | 0.1826 | 0.8468 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2906 | 0.6411 | 0.5999 | 0.6139 | 0.8434 | 0.8205 | 0.8545 | 0.0118 | 0.5310 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7098 | 0.8104 | 0.8131 | 0.8667 | 0.9365 | 0.9382 | 0.9665 | 0.2554 | 0.8544 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.7366 | 0.8231 | 0.8235 | 0.8999 | 0.9401 | 0.9400 | 0.8865 | 0.1905 | 0.7050 |
