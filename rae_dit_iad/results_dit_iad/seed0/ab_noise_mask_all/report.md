# RAE-DiT IAD results (ab_noise_mask_all)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5172 | 0.5739 | 0.5957 | 0.6759 | 0.7008 | 0.7178 | 0.5049 | 0.0639 | 0.2957 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2866 | 0.3756 | 0.3527 | 0.5667 | 0.6158 | 0.6036 | 0.1845 | 0.0124 | 0.0178 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4787 | 0.6582 | 0.6635 | 0.6883 | 0.8031 | 0.8024 | 0.7535 | 0.1648 | 0.5696 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.3848 | 0.4081 | 0.4390 | 0.7065 | 0.7285 | 0.7445 | 0.3605 | 0.0178 | 0.1155 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.0795 | 0.3387 | 0.3178 | 0.5712 | 0.6860 | 0.6931 | 0.0928 | 0.0083 | 0.0011 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.7865 | 0.9506 | 0.9727 | 0.9190 | 0.9842 | 0.9918 | 0.8602 | 0.1875 | 0.7846 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.7377 | 0.8262 | 0.8822 | 0.8689 | 0.9205 | 0.9488 | 0.8733 | 0.0932 | 0.7417 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2565 | 0.3960 | 0.3642 | 0.5929 | 0.6896 | 0.6453 | 0.0282 | 0.0035 | 0.0000 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.5288 | 0.6190 | 0.6725 | 0.7708 | 0.8246 | 0.8559 | 0.7190 | 0.0779 | 0.4850 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5304 | 0.9475 | 0.9154 | 0.7051 | 0.9707 | 0.9516 | 0.9644 | 0.5433 | 0.9218 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6207 | 0.5782 | 0.5518 | 0.7594 | 0.7185 | 0.7188 | 0.2029 | 0.0127 | 0.0000 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.4643 | 0.5046 | 0.5250 | 0.6179 | 0.6347 | 0.6484 | 0.0961 | 0.0111 | 0.0057 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0092 | 0.5899 | 0.5288 | 0.5347 | 0.8419 | 0.8146 | 0.8795 | 0.0376 | 0.6002 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2716 | 0.3962 | 0.3644 | 0.6233 | 0.7227 | 0.6988 | 0.2050 | 0.0015 | 0.0000 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.3140 | 0.2879 | 0.2691 | 0.6331 | 0.6442 | 0.6241 | 0.4027 | 0.0019 | 0.0676 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.7313 | 0.5192 | 0.5521 | 0.6264 | 0.5539 | 0.5524 | 0.6899 | 0.1319 | 0.4691 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2050 | 0.1688 | 0.1654 | 0.2866 | 0.2623 | 0.2620 | 0.3935 | 0.0360 | 0.0878 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.4925 | 0.5075 | 0.5392 | 0.4389 | 0.4165 | 0.4686 | 0.4464 | 0.0409 | 0.1358 |
