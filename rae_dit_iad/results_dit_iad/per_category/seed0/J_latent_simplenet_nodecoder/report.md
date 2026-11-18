# RAE-DiT IAD results (J_latent_simplenet_nodecoder)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6420 | 0.7520 | 0.7345 | 0.8320 | 0.8789 | 0.8676 | 0.7283 | 0.0477 | 0.3848 |
| single | 0.1 | 1 | best | latent_cos | 0.8383 | 0.8148 | 0.8254 | 0.9209 | 0.9004 | 0.9043 | 0.8691 | 0.1644 | 0.5924 |
| single | 0.1 | 1 | best | pixel | 0.3986 | 0.5746 | 0.5732 | 0.7171 | 0.8239 | 0.8258 | 0.5959 | 0.0110 | 0.2958 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5084 | 0.5108 | 0.5566 | 0.8219 | 0.8065 | 0.8386 | 0.3416 | 0.0114 | 0.0688 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.9563 | 0.9446 | 0.9482 | 0.9873 | 0.9485 | 0.9551 | 0.9624 | 0.3004 | 0.8119 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8503 | 0.9699 | 0.9671 | 0.9531 | 0.9913 | 0.9908 | 0.8315 | 0.0923 | 0.6500 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.2888 | 0.6385 | 0.5899 | 0.6123 | 0.8414 | 0.8130 | 0.8503 | 0.0106 | 0.5229 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.7204 | 0.6850 | 0.7026 | 0.8546 | 0.8523 | 0.8535 | 0.7757 | 0.0284 | 0.3728 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.4337 | 0.5341 | 0.5019 | 0.7109 | 0.7665 | 0.7445 | 0.6251 | 0.0030 | 0.1197 |
