# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | joint_best.pt | latent_cos | 0.6395 | 0.7486 | 0.7850 | 0.7627 | 0.8211 | 0.8449 | 0.8568 | 0.2387 | 0.7419 |
| single | 0.5 | 1 | joint_best.pt | pixel | 0.4495 | 0.5990 | 0.5878 | 0.6494 | 0.7401 | 0.7323 | 0.6022 | 0.0744 | 0.3546 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | joint_best.pt | pixel | 0.6400 | 0.6505 | 0.6665 | 0.8844 | 0.8749 | 0.8894 | 0.4559 | 0.0166 | 0.1243 |
| carpet | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.7139 | 0.8046 | 0.8399 | 0.9030 | 0.9408 | 0.9533 | 0.9388 | 0.4162 | 0.8045 |
| grid | single | 0.5 | 1 | joint_best.pt | pixel | 0.4628 | 0.4996 | 0.4804 | 0.7215 | 0.7608 | 0.7421 | 0.5593 | 0.0078 | 0.1886 |
| grid | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8613 | 0.8972 | 0.9766 | 0.9511 | 0.9593 | 0.9917 | 0.9582 | 0.1329 | 0.8852 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | pixel | 0.4518 | 0.8718 | 0.8179 | 0.6068 | 0.9283 | 0.8917 | 0.8150 | 0.3018 | 0.8319 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.7189 | 0.8096 | 0.8664 | 0.8340 | 0.8911 | 0.9305 | 0.9624 | 0.4431 | 0.8689 |
| screw | single | 0.5 | 1 | joint_best.pt | pixel | 0.2874 | 0.5780 | 0.5532 | 0.6153 | 0.7812 | 0.7578 | 0.8619 | 0.0129 | 0.5542 |
| screw | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.4421 | 0.7051 | 0.7012 | 0.6872 | 0.8609 | 0.8710 | 0.9421 | 0.1481 | 0.7896 |
| transistor | single | 0.5 | 1 | joint_best.pt | pixel | 0.4054 | 0.3950 | 0.4212 | 0.4187 | 0.3555 | 0.3805 | 0.3190 | 0.0330 | 0.0740 |
| transistor | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.4612 | 0.5267 | 0.5408 | 0.4384 | 0.4536 | 0.4783 | 0.4825 | 0.0532 | 0.3611 |
