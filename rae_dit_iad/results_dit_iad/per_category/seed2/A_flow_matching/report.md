# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.8808 | 0.9705 | 0.9792 | 0.9239 | 0.9757 | 0.9845 | 0.9691 | 0.4381 | 0.8863 |
| single | 0.5 | 1 | best | pixel | 0.4274 | 0.5999 | 0.5885 | 0.6404 | 0.7572 | 0.7554 | 0.6381 | 0.1165 | 0.4277 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.3880 | 0.4270 | 0.4555 | 0.7525 | 0.7509 | 0.7770 | 0.4128 | 0.0132 | 0.1125 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9482 | 0.9928 | 0.9960 | 0.9856 | 0.9978 | 0.9988 | 0.9809 | 0.4810 | 0.9075 |
| grid | single | 0.5 | 1 | grid_best.pt | pixel | 0.4536 | 0.5029 | 0.4653 | 0.7156 | 0.7664 | 0.7418 | 0.6560 | 0.0329 | 0.3683 |
| grid | single | 0.5 | 1 | grid_best.pt | latent_cos | 0.9850 | 0.9983 | 1.0000 | 0.9948 | 0.9994 | 1.0000 | 0.9827 | 0.2099 | 0.9378 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | pixel | 0.4711 | 0.9379 | 0.8639 | 0.6163 | 0.9700 | 0.9308 | 0.8380 | 0.4433 | 0.8376 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | latent_cos | 0.9068 | 0.9704 | 0.9900 | 0.9518 | 0.9745 | 0.9937 | 0.9934 | 0.7615 | 0.9595 |
| screw | single | 0.5 | 1 | screw_best.pt | pixel | 0.3859 | 0.7014 | 0.6971 | 0.6832 | 0.8624 | 0.8676 | 0.9062 | 0.0483 | 0.6849 |
| screw | single | 0.5 | 1 | screw_best.pt | latent_cos | 0.6962 | 0.9488 | 0.9494 | 0.8382 | 0.9822 | 0.9811 | 0.9740 | 0.3363 | 0.8855 |
| transistor | single | 0.5 | 1 | transistor_best.pt | pixel | 0.4383 | 0.4304 | 0.4608 | 0.4343 | 0.4365 | 0.4597 | 0.3776 | 0.0448 | 0.1350 |
| transistor | single | 0.5 | 1 | transistor_best.pt | latent_cos | 0.8679 | 0.9425 | 0.9604 | 0.8493 | 0.9245 | 0.9490 | 0.9144 | 0.4016 | 0.7413 |
