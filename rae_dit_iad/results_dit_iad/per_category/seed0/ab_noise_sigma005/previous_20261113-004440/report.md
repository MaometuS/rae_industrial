# RAE-DiT IAD results (ab_noise_sigma005)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.05 | 1 | best | disc | 0.5676 | 0.5939 | 0.5852 | 0.7324 | 0.7517 | 0.7502 | 0.7533 | 0.1216 | 0.3135 |
| single | 0.05 | 1 | best | latent_cos | 0.3622 | 0.4778 | 0.4564 | 0.5992 | 0.6760 | 0.6585 | 0.4646 | 0.0415 | 0.2711 |
| single | 0.05 | 1 | best | pixel | 0.4766 | 0.6920 | 0.6973 | 0.7087 | 0.8113 | 0.8230 | 0.8227 | 0.2478 | 0.6130 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.05 | 1 | carpet_best.pt | pixel | 0.3612 | 0.4125 | 0.4145 | 0.6783 | 0.7206 | 0.7203 | 0.5708 | 0.0349 | 0.2124 |
| carpet | single | 0.05 | 1 | carpet_best.pt | latent_cos | 0.3929 | 0.4980 | 0.4819 | 0.7353 | 0.8038 | 0.7964 | 0.6598 | 0.0763 | 0.4003 |
| carpet | single | 0.05 | 1 | carpet_best.pt | disc | 0.5907 | 0.4318 | 0.4133 | 0.8282 | 0.7566 | 0.7453 | 0.6072 | 0.0182 | 0.0956 |
| grid | single | 0.05 | 1 | grid_best.pt | pixel | 0.7235 | 0.8020 | 0.8864 | 0.8425 | 0.9212 | 0.9530 | 0.9329 | 0.2102 | 0.8603 |
| grid | single | 0.05 | 1 | grid_best.pt | latent_cos | 0.0693 | 0.2840 | 0.2206 | 0.5366 | 0.6553 | 0.6055 | 0.0751 | 0.0036 | 0.0040 |
| grid | single | 0.05 | 1 | grid_best.pt | disc | 0.4319 | 0.8413 | 0.8354 | 0.6743 | 0.9343 | 0.9357 | 0.7775 | 0.1005 | 0.5379 |
| hazelnut | single | 0.05 | 1 | hazelnut_best.pt | pixel | 0.4654 | 0.9186 | 0.8636 | 0.6843 | 0.9615 | 0.9347 | 0.9519 | 0.5553 | 0.8650 |
| hazelnut | single | 0.05 | 1 | hazelnut_best.pt | latent_cos | 0.5757 | 0.6432 | 0.6396 | 0.6753 | 0.7231 | 0.7207 | 0.8683 | 0.0972 | 0.7632 |
| hazelnut | single | 0.05 | 1 | hazelnut_best.pt | disc | 0.5332 | 0.5704 | 0.5561 | 0.7044 | 0.7431 | 0.7403 | 0.9053 | 0.1703 | 0.5147 |
| screw | single | 0.05 | 1 | screw_best.pt | pixel | 0.0020 | 0.7838 | 0.7042 | 0.5341 | 0.9189 | 0.8705 | 0.9305 | 0.2509 | 0.7439 |
| screw | single | 0.05 | 1 | screw_best.pt | latent_cos | 0.4774 | 0.5163 | 0.5030 | 0.7434 | 0.7759 | 0.7654 | 0.5136 | 0.0026 | 0.1430 |
| screw | single | 0.05 | 1 | screw_best.pt | disc | 0.6006 | 0.4782 | 0.4718 | 0.7989 | 0.7235 | 0.7176 | 0.6419 | 0.0032 | 0.1228 |
| transistor | single | 0.05 | 1 | transistor_best.pt | pixel | 0.8308 | 0.5429 | 0.6179 | 0.8046 | 0.5342 | 0.6363 | 0.7273 | 0.1879 | 0.3836 |
| transistor | single | 0.05 | 1 | transistor_best.pt | latent_cos | 0.2958 | 0.4475 | 0.4367 | 0.3053 | 0.4216 | 0.4046 | 0.2060 | 0.0276 | 0.0448 |
| transistor | single | 0.05 | 1 | transistor_best.pt | disc | 0.6817 | 0.6479 | 0.6496 | 0.6565 | 0.6011 | 0.6121 | 0.8347 | 0.3159 | 0.2963 |
