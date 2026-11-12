# RAE-DiT IAD results (E_image_l1_latent_decfinetune)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6589 | 0.6780 | 0.6743 | 0.7718 | 0.7939 | 0.7913 | 0.7680 | 0.1645 | 0.4099 |
| single | 0.1 | 1 | best | latent_cos | 0.6567 | 0.6367 | 0.6965 | 0.7888 | 0.7425 | 0.7762 | 0.8183 | 0.1564 | 0.5815 |
| single | 0.1 | 1 | best | pixel | 0.5038 | 0.7090 | 0.6852 | 0.6947 | 0.8202 | 0.8047 | 0.8409 | 0.2450 | 0.6454 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4097 | 0.4081 | 0.4113 | 0.6889 | 0.7230 | 0.7062 | 0.5894 | 0.0206 | 0.1520 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.6505 | 0.5349 | 0.6457 | 0.8736 | 0.7691 | 0.8292 | 0.8721 | 0.0650 | 0.7440 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9097 | 0.6156 | 0.6176 | 0.9727 | 0.8559 | 0.8605 | 0.6718 | 0.0287 | 0.2415 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.8338 | 0.9273 | 0.9683 | 0.9181 | 0.9687 | 0.9884 | 0.9746 | 0.3525 | 0.9330 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.6955 | 0.8997 | 0.9273 | 0.8875 | 0.9630 | 0.9735 | 0.9565 | 0.1384 | 0.8358 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5054 | 0.8805 | 0.8764 | 0.7218 | 0.9563 | 0.9539 | 0.7936 | 0.0951 | 0.5733 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.5521 | 0.9271 | 0.9050 | 0.6777 | 0.9586 | 0.9454 | 0.9702 | 0.5666 | 0.9201 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.7461 | 0.6307 | 0.7189 | 0.8505 | 0.7240 | 0.7969 | 0.9604 | 0.4506 | 0.6746 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.7600 | 0.7439 | 0.7211 | 0.8540 | 0.8667 | 0.8557 | 0.9674 | 0.5482 | 0.7490 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0275 | 0.6844 | 0.5319 | 0.5382 | 0.8638 | 0.7818 | 0.9084 | 0.1155 | 0.6991 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.4249 | 0.4782 | 0.4769 | 0.7021 | 0.7361 | 0.7205 | 0.5502 | 0.0070 | 0.3113 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5827 | 0.5757 | 0.6009 | 0.7918 | 0.7833 | 0.7972 | 0.6538 | 0.0040 | 0.2269 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.6958 | 0.5979 | 0.6096 | 0.6506 | 0.5867 | 0.6017 | 0.7619 | 0.1698 | 0.5229 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.7667 | 0.6400 | 0.7137 | 0.6303 | 0.5203 | 0.5607 | 0.7524 | 0.1207 | 0.3418 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5367 | 0.5742 | 0.5554 | 0.5189 | 0.5072 | 0.4890 | 0.7533 | 0.1464 | 0.2586 |
