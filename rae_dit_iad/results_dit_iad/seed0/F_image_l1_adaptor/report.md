# RAE-DiT IAD results (F_image_l1_adaptor)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5777 | 0.5052 | 0.4910 | 0.7583 | 0.6961 | 0.6867 | 0.7147 | 0.0739 | 0.2763 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3721 | 0.4494 | 0.4501 | 0.6078 | 0.6606 | 0.6600 | 0.5292 | 0.0328 | 0.3376 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4215 | 0.5840 | 0.5592 | 0.6579 | 0.7597 | 0.7454 | 0.7807 | 0.1656 | 0.5860 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.3507 | 0.5329 | 0.5088 | 0.6987 | 0.8306 | 0.8207 | 0.5753 | 0.0981 | 0.3209 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3415 | 0.5598 | 0.5409 | 0.7030 | 0.8123 | 0.8118 | 0.0785 | 0.0083 | 0.0010 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.6312 | 0.1609 | 0.1501 | 0.8764 | 0.6045 | 0.5987 | 0.6972 | 0.0247 | 0.1672 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.7845 | 0.8839 | 0.8897 | 0.9009 | 0.9499 | 0.9551 | 0.8600 | 0.1114 | 0.7435 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.1454 | 0.2991 | 0.2523 | 0.5642 | 0.6693 | 0.6314 | 0.4002 | 0.0280 | 0.1884 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.5906 | 0.6750 | 0.6767 | 0.8030 | 0.8639 | 0.8725 | 0.5401 | 0.0068 | 0.0271 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.3389 | 0.9082 | 0.8600 | 0.5885 | 0.9533 | 0.9258 | 0.9387 | 0.4803 | 0.8587 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4739 | 0.5343 | 0.5371 | 0.6210 | 0.7016 | 0.7039 | 0.8054 | 0.0524 | 0.5936 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.6932 | 0.8564 | 0.8414 | 0.7958 | 0.9274 | 0.9160 | 0.9063 | 0.2410 | 0.6854 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0000 | 0.1041 | 0.0535 | 0.5340 | 0.5633 | 0.5432 | 0.8668 | 0.0135 | 0.5837 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5944 | 0.5026 | 0.5300 | 0.7990 | 0.7533 | 0.7692 | 0.9192 | 0.0164 | 0.7213 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.3121 | 0.4124 | 0.4081 | 0.6377 | 0.7042 | 0.6897 | 0.6908 | 0.0088 | 0.2808 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.6333 | 0.4908 | 0.4837 | 0.5676 | 0.5016 | 0.4821 | 0.6625 | 0.1246 | 0.4232 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3054 | 0.3512 | 0.3900 | 0.3517 | 0.3664 | 0.3836 | 0.4428 | 0.0587 | 0.1837 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.6615 | 0.4213 | 0.3788 | 0.6785 | 0.3806 | 0.3567 | 0.7389 | 0.0884 | 0.2211 |
