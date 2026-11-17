# RAE-DiT IAD results (ab_noise_sigma005)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.05 | 1 | best | disc | 0.6440 | 0.6710 | 0.6700 | 0.7639 | 0.7777 | 0.7803 | 0.7227 | 0.1136 | 0.3279 |
| single | 0.05 | 1 | best | latent_cos | 0.2644 | 0.3774 | 0.3394 | 0.5630 | 0.6088 | 0.5973 | 0.2731 | 0.0134 | 0.1291 |
| single | 0.05 | 1 | best | pixel | 0.5149 | 0.7149 | 0.6955 | 0.7184 | 0.8222 | 0.8172 | 0.7759 | 0.2005 | 0.6102 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.05 | 1 | carpet_best.pt | pixel | 0.4117 | 0.5369 | 0.5104 | 0.7527 | 0.8094 | 0.8010 | 0.3799 | 0.0180 | 0.1290 |
| carpet | single | 0.05 | 1 | carpet_best.pt | latent_cos | 0.0634 | 0.2022 | 0.1657 | 0.5669 | 0.6322 | 0.6154 | 0.0256 | 0.0081 | 0.0008 |
| carpet | single | 0.05 | 1 | carpet_best.pt | disc | 0.9065 | 0.7460 | 0.7528 | 0.9712 | 0.9204 | 0.9237 | 0.5744 | 0.0171 | 0.1330 |
| grid | single | 0.05 | 1 | grid_best.pt | pixel | 0.7711 | 0.8396 | 0.9290 | 0.8878 | 0.9276 | 0.9719 | 0.9415 | 0.2118 | 0.8735 |
| grid | single | 0.05 | 1 | grid_best.pt | latent_cos | 0.1345 | 0.3475 | 0.2749 | 0.5548 | 0.6466 | 0.6098 | 0.0137 | 0.0035 | 0.0000 |
| grid | single | 0.05 | 1 | grid_best.pt | disc | 0.4937 | 0.6784 | 0.6909 | 0.7202 | 0.8296 | 0.8488 | 0.7138 | 0.0302 | 0.3939 |
| hazelnut | single | 0.05 | 1 | hazelnut_best.pt | pixel | 0.6446 | 0.9564 | 0.9236 | 0.7767 | 0.9748 | 0.9557 | 0.9711 | 0.5710 | 0.9260 |
| hazelnut | single | 0.05 | 1 | hazelnut_best.pt | latent_cos | 0.3171 | 0.5446 | 0.5725 | 0.5613 | 0.6861 | 0.7365 | 0.1143 | 0.0114 | 0.0000 |
| hazelnut | single | 0.05 | 1 | hazelnut_best.pt | disc | 0.7182 | 0.7418 | 0.7268 | 0.8345 | 0.8532 | 0.8423 | 0.9443 | 0.3728 | 0.6838 |
| screw | single | 0.05 | 1 | screw_best.pt | pixel | 0.0164 | 0.6419 | 0.5362 | 0.5363 | 0.8458 | 0.7896 | 0.8798 | 0.0598 | 0.6070 |
| screw | single | 0.05 | 1 | screw_best.pt | latent_cos | 0.7022 | 0.5308 | 0.5155 | 0.8820 | 0.7779 | 0.7631 | 0.8524 | 0.0094 | 0.5266 |
| screw | single | 0.05 | 1 | screw_best.pt | disc | 0.5655 | 0.5856 | 0.5776 | 0.7715 | 0.7737 | 0.7754 | 0.6310 | 0.0031 | 0.1473 |
| transistor | single | 0.05 | 1 | transistor_best.pt | pixel | 0.7304 | 0.5996 | 0.5783 | 0.6385 | 0.5532 | 0.5676 | 0.7071 | 0.1416 | 0.5153 |
| transistor | single | 0.05 | 1 | transistor_best.pt | latent_cos | 0.1046 | 0.2617 | 0.1683 | 0.2501 | 0.3012 | 0.2619 | 0.3596 | 0.0344 | 0.1181 |
| transistor | single | 0.05 | 1 | transistor_best.pt | disc | 0.5363 | 0.6033 | 0.6021 | 0.5218 | 0.5114 | 0.5111 | 0.7501 | 0.1449 | 0.2813 |
