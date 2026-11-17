# RAE-DiT IAD results (D_image_l1_latent)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6650 | 0.7481 | 0.7441 | 0.7733 | 0.8220 | 0.8209 | 0.7722 | 0.1461 | 0.4303 |
| single | 0.1 | 1 | best | latent_cos | 0.4068 | 0.4493 | 0.3965 | 0.6105 | 0.6627 | 0.6302 | 0.7722 | 0.0445 | 0.4842 |
| single | 0.1 | 1 | best | pixel | 0.5602 | 0.6699 | 0.6761 | 0.7456 | 0.7963 | 0.8093 | 0.7383 | 0.1472 | 0.5222 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5128 | 0.5229 | 0.5461 | 0.7942 | 0.8023 | 0.8117 | 0.4202 | 0.0242 | 0.1318 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.5815 | 0.4908 | 0.3752 | 0.8163 | 0.7949 | 0.7293 | 0.9180 | 0.0955 | 0.7660 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9109 | 0.9430 | 0.9394 | 0.9712 | 0.9844 | 0.9835 | 0.7261 | 0.0420 | 0.4276 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.7076 | 0.7118 | 0.8212 | 0.8722 | 0.8792 | 0.9234 | 0.8478 | 0.0649 | 0.6777 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.3434 | 0.3317 | 0.3467 | 0.6403 | 0.6639 | 0.6705 | 0.7074 | 0.0113 | 0.2417 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5664 | 0.8563 | 0.8505 | 0.7600 | 0.9448 | 0.9411 | 0.7872 | 0.0838 | 0.5393 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.7536 | 0.9368 | 0.9021 | 0.8564 | 0.9674 | 0.9477 | 0.9656 | 0.5277 | 0.9073 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.3996 | 0.5671 | 0.4786 | 0.5697 | 0.7534 | 0.6520 | 0.8123 | 0.0530 | 0.4861 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.7536 | 0.7779 | 0.7721 | 0.8568 | 0.8744 | 0.8693 | 0.9553 | 0.4124 | 0.7282 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.1931 | 0.6641 | 0.6104 | 0.6097 | 0.8648 | 0.8318 | 0.8361 | 0.0238 | 0.4983 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.5331 | 0.5880 | 0.5573 | 0.7549 | 0.7984 | 0.7871 | 0.9341 | 0.0191 | 0.7835 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5602 | 0.5462 | 0.5462 | 0.7656 | 0.7657 | 0.7699 | 0.6413 | 0.0033 | 0.1654 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.6338 | 0.5138 | 0.5008 | 0.5955 | 0.4679 | 0.5318 | 0.6218 | 0.0953 | 0.3959 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.1763 | 0.2687 | 0.2246 | 0.2714 | 0.3028 | 0.3122 | 0.4894 | 0.0436 | 0.1435 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5342 | 0.6171 | 0.6121 | 0.5132 | 0.5408 | 0.5405 | 0.7511 | 0.1892 | 0.2910 |
