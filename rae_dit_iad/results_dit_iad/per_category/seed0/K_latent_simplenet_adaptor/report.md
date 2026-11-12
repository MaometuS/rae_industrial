# RAE-DiT IAD results (K_latent_simplenet_adaptor)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.5858 | 0.6759 | 0.6734 | 0.7264 | 0.7815 | 0.7893 | 0.7506 | 0.1095 | 0.4150 |
| single | 0.1 | 1 | best | latent_cos | 0.8029 | 0.7690 | 0.7869 | 0.8814 | 0.8708 | 0.8809 | 0.7822 | 0.1960 | 0.3581 |
| single | 0.1 | 1 | best | pixel | 0.4216 | 0.5671 | 0.5464 | 0.6407 | 0.7299 | 0.7277 | 0.5708 | 0.0739 | 0.3333 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4980 | 0.5072 | 0.5437 | 0.8155 | 0.8032 | 0.8313 | 0.3393 | 0.0113 | 0.0678 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.9205 | 0.8732 | 0.8892 | 0.9773 | 0.9622 | 0.9677 | 0.7244 | 0.0424 | 0.3421 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8315 | 0.8234 | 0.8226 | 0.9448 | 0.9450 | 0.9423 | 0.7217 | 0.0445 | 0.4228 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.3968 | 0.4252 | 0.3943 | 0.6897 | 0.7248 | 0.6892 | 0.5273 | 0.0073 | 0.2108 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.7937 | 0.8070 | 0.8179 | 0.8872 | 0.9163 | 0.9184 | 0.8245 | 0.0396 | 0.4642 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.4762 | 0.9081 | 0.8889 | 0.7049 | 0.9641 | 0.9607 | 0.7078 | 0.0365 | 0.4065 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.4832 | 0.8893 | 0.7943 | 0.6469 | 0.9475 | 0.8945 | 0.8095 | 0.3068 | 0.7847 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.8339 | 0.7604 | 0.7779 | 0.9064 | 0.8747 | 0.8789 | 0.8848 | 0.5010 | 0.4950 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.5418 | 0.7493 | 0.7704 | 0.6872 | 0.8344 | 0.8633 | 0.9570 | 0.3495 | 0.8315 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.3095 | 0.6442 | 0.6069 | 0.6242 | 0.8446 | 0.8259 | 0.8534 | 0.0114 | 0.5328 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.6011 | 0.5188 | 0.5231 | 0.7805 | 0.7297 | 0.7295 | 0.6226 | 0.0032 | 0.1395 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.4585 | 0.5079 | 0.4972 | 0.6976 | 0.7260 | 0.7249 | 0.6122 | 0.0029 | 0.1030 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.4204 | 0.3696 | 0.3929 | 0.4270 | 0.3292 | 0.3977 | 0.3244 | 0.0330 | 0.0704 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.8654 | 0.8858 | 0.9267 | 0.8555 | 0.8711 | 0.9099 | 0.8547 | 0.3937 | 0.3496 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.6212 | 0.3908 | 0.3879 | 0.5973 | 0.4380 | 0.4550 | 0.7542 | 0.1139 | 0.3110 |
