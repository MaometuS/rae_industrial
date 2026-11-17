# RAE-DiT IAD results (E_image_l1_latent_decfinetune)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6673 | 0.6870 | 0.6868 | 0.7737 | 0.7959 | 0.7949 | 0.7242 | 0.1263 | 0.3540 |
| single | 0.1 | 1 | best | latent_cos | 0.6909 | 0.5293 | 0.5604 | 0.7979 | 0.6742 | 0.6988 | 0.9015 | 0.1418 | 0.6811 |
| single | 0.1 | 1 | best | pixel | 0.4958 | 0.7149 | 0.6979 | 0.6966 | 0.8217 | 0.8072 | 0.8217 | 0.2300 | 0.6319 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4274 | 0.4021 | 0.4330 | 0.6977 | 0.7167 | 0.7136 | 0.5311 | 0.0180 | 0.1270 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.9386 | 0.6396 | 0.7315 | 0.9815 | 0.8597 | 0.9016 | 0.9503 | 0.1914 | 0.8385 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9153 | 0.7014 | 0.7135 | 0.9748 | 0.9020 | 0.9044 | 0.4578 | 0.0134 | 0.0327 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.8170 | 0.8906 | 0.9749 | 0.9182 | 0.9496 | 0.9908 | 0.9698 | 0.3453 | 0.9249 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.8087 | 0.4302 | 0.4536 | 0.9178 | 0.6823 | 0.7090 | 0.9263 | 0.0527 | 0.7296 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5647 | 0.8956 | 0.8897 | 0.7563 | 0.9617 | 0.9594 | 0.8024 | 0.1059 | 0.5904 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.5096 | 0.9518 | 0.9093 | 0.6779 | 0.9730 | 0.9499 | 0.9694 | 0.5647 | 0.9242 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.6329 | 0.5021 | 0.5125 | 0.7852 | 0.6444 | 0.6751 | 0.9540 | 0.2916 | 0.6583 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.7582 | 0.6500 | 0.6257 | 0.8529 | 0.7855 | 0.7679 | 0.9266 | 0.3178 | 0.6084 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0123 | 0.7118 | 0.5696 | 0.5360 | 0.8526 | 0.7817 | 0.8998 | 0.0704 | 0.6674 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.4190 | 0.5786 | 0.5733 | 0.6885 | 0.7973 | 0.8024 | 0.9301 | 0.0695 | 0.7333 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5577 | 0.5589 | 0.5893 | 0.7622 | 0.7844 | 0.8017 | 0.6818 | 0.0061 | 0.2366 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7125 | 0.6183 | 0.6029 | 0.6534 | 0.6167 | 0.6001 | 0.7384 | 0.1517 | 0.5162 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.6552 | 0.4958 | 0.5312 | 0.6164 | 0.3873 | 0.4061 | 0.7468 | 0.1040 | 0.4456 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5404 | 0.6292 | 0.6158 | 0.5224 | 0.5458 | 0.5410 | 0.7525 | 0.1885 | 0.3020 |
