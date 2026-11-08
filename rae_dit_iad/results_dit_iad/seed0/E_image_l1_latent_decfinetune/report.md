# RAE-DiT IAD results (E_image_l1_latent_decfinetune)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5138 | 0.4996 | 0.5156 | 0.7020 | 0.7054 | 0.7193 | 0.7138 | 0.0657 | 0.2344 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5488 | 0.6225 | 0.6535 | 0.7113 | 0.7464 | 0.7725 | 0.8760 | 0.1599 | 0.6269 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4946 | 0.6983 | 0.6828 | 0.6963 | 0.8126 | 0.7947 | 0.8124 | 0.2149 | 0.6103 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.4607 | 0.4478 | 0.4823 | 0.7082 | 0.7179 | 0.7146 | 0.5058 | 0.0158 | 0.1149 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7173 | 0.6693 | 0.7343 | 0.9064 | 0.8654 | 0.9023 | 0.9424 | 0.1903 | 0.8419 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.6898 | 0.4410 | 0.4707 | 0.9054 | 0.7819 | 0.8029 | 0.7453 | 0.0354 | 0.1860 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.8321 | 0.8805 | 0.9683 | 0.9272 | 0.9432 | 0.9889 | 0.9542 | 0.2804 | 0.9019 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5439 | 0.7093 | 0.7577 | 0.8049 | 0.8753 | 0.9000 | 0.9379 | 0.0965 | 0.7806 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.3843 | 0.6282 | 0.6458 | 0.7215 | 0.8100 | 0.8438 | 0.5254 | 0.0066 | 0.0277 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.4586 | 0.9532 | 0.9236 | 0.6493 | 0.9741 | 0.9556 | 0.9719 | 0.5705 | 0.9266 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6459 | 0.6379 | 0.7011 | 0.7903 | 0.7130 | 0.7510 | 0.9283 | 0.3786 | 0.6047 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7107 | 0.5464 | 0.5707 | 0.8322 | 0.7161 | 0.7258 | 0.8982 | 0.1778 | 0.5829 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0115 | 0.6376 | 0.4960 | 0.5350 | 0.8422 | 0.7613 | 0.8803 | 0.0535 | 0.6059 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3894 | 0.4612 | 0.4396 | 0.6857 | 0.7253 | 0.7122 | 0.8321 | 0.0253 | 0.5590 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.3382 | 0.4769 | 0.4757 | 0.6509 | 0.7730 | 0.7808 | 0.6652 | 0.0034 | 0.1899 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.7100 | 0.5725 | 0.5438 | 0.6621 | 0.5858 | 0.5533 | 0.7497 | 0.1544 | 0.5021 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4473 | 0.6350 | 0.6346 | 0.3692 | 0.5532 | 0.5973 | 0.7393 | 0.1088 | 0.3481 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.4458 | 0.4054 | 0.4150 | 0.4003 | 0.4461 | 0.4430 | 0.7350 | 0.1052 | 0.1853 |
