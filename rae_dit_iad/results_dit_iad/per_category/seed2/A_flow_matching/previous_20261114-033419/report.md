# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.8811 | 0.9697 | 0.9788 | 0.9248 | 0.9753 | 0.9841 | 0.9690 | 0.4384 | 0.8859 |
| single | 0.5 | 1 | best | pixel | 0.4334 | 0.6068 | 0.5929 | 0.6433 | 0.7600 | 0.7564 | 0.6384 | 0.1153 | 0.4262 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4205 | 0.4530 | 0.4839 | 0.7672 | 0.7668 | 0.7890 | 0.4159 | 0.0132 | 0.1165 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9478 | 0.9924 | 0.9956 | 0.9854 | 0.9977 | 0.9987 | 0.9810 | 0.4845 | 0.9074 |
| grid | single | 0.5 | 1 | grid_best.pt | pixel | 0.4561 | 0.5163 | 0.4712 | 0.7158 | 0.7739 | 0.7426 | 0.6571 | 0.0326 | 0.3661 |
| grid | single | 0.5 | 1 | grid_best.pt | latent_cos | 0.9841 | 1.0000 | 1.0000 | 0.9944 | 1.0000 | 1.0000 | 0.9824 | 0.2094 | 0.9371 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | pixel | 0.4761 | 0.9368 | 0.8557 | 0.6178 | 0.9690 | 0.9265 | 0.8389 | 0.4396 | 0.8360 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | latent_cos | 0.9082 | 0.9725 | 0.9914 | 0.9526 | 0.9783 | 0.9948 | 0.9934 | 0.7625 | 0.9593 |
| screw | single | 0.5 | 1 | screw_best.pt | pixel | 0.3876 | 0.6965 | 0.6889 | 0.6834 | 0.8588 | 0.8626 | 0.9040 | 0.0464 | 0.6795 |
| screw | single | 0.5 | 1 | screw_best.pt | latent_cos | 0.6911 | 0.9434 | 0.9479 | 0.8361 | 0.9803 | 0.9805 | 0.9738 | 0.3381 | 0.8840 |
| transistor | single | 0.5 | 1 | transistor_best.pt | pixel | 0.4267 | 0.4317 | 0.4646 | 0.4323 | 0.4314 | 0.4614 | 0.3760 | 0.0447 | 0.1330 |
| transistor | single | 0.5 | 1 | transistor_best.pt | latent_cos | 0.8742 | 0.9404 | 0.9592 | 0.8554 | 0.9203 | 0.9466 | 0.9143 | 0.3975 | 0.7415 |
