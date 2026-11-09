# RAE-DiT IAD results (K2_latent_adaptor_sigma05_15)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | joint_best.pt | disc | 0.8877 | 0.8182 | 0.8421 | 0.9126 | 0.8709 | 0.8847 | 0.8380 | 0.1875 | 0.7618 |
| single | 1.0 | 1 | joint_best.pt | latent_cos | 0.8780 | 0.8354 | 0.8603 | 0.9124 | 0.8790 | 0.8909 | 0.9294 | 0.2094 | 0.6647 |
| single | 1.0 | 1 | joint_best.pt | pixel | 0.4106 | 0.5737 | 0.5461 | 0.6372 | 0.7324 | 0.7263 | 0.5709 | 0.0753 | 0.3341 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | joint_best.pt | pixel | 0.5181 | 0.5205 | 0.5650 | 0.8263 | 0.8098 | 0.8455 | 0.3437 | 0.0114 | 0.0693 |
| carpet | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.9771 | 0.9807 | 0.9864 | 0.9933 | 0.9942 | 0.9958 | 0.9066 | 0.0998 | 0.5892 |
| carpet | single | 1.0 | 1 | joint_best.pt | disc | 0.9968 | 0.7717 | 0.8315 | 0.9990 | 0.9078 | 0.9376 | 0.7828 | 0.1037 | 0.7444 |
| grid | single | 1.0 | 1 | joint_best.pt | pixel | 0.3977 | 0.4336 | 0.3910 | 0.6904 | 0.7285 | 0.6871 | 0.5243 | 0.0072 | 0.2099 |
| grid | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.9983 | 0.9883 | 0.9942 | 0.9994 | 0.9960 | 0.9980 | 0.9424 | 0.1096 | 0.7821 |
| grid | single | 1.0 | 1 | joint_best.pt | disc | 0.9850 | 0.9833 | 0.9841 | 0.9944 | 0.9939 | 0.9942 | 0.9341 | 0.1925 | 0.8637 |
| hazelnut | single | 1.0 | 1 | joint_best.pt | pixel | 0.4850 | 0.8954 | 0.7989 | 0.6464 | 0.9504 | 0.8975 | 0.8023 | 0.3136 | 0.7822 |
| hazelnut | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.9393 | 0.8843 | 0.9179 | 0.9695 | 0.9338 | 0.9505 | 0.9770 | 0.4482 | 0.8489 |
| hazelnut | single | 1.0 | 1 | joint_best.pt | disc | 0.9918 | 0.9668 | 0.9818 | 0.9962 | 0.9831 | 0.9906 | 0.9554 | 0.3569 | 0.9306 |
| screw | single | 1.0 | 1 | joint_best.pt | pixel | 0.2306 | 0.6231 | 0.5638 | 0.5911 | 0.8307 | 0.7962 | 0.8506 | 0.0103 | 0.5231 |
| screw | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.6438 | 0.5845 | 0.6393 | 0.8174 | 0.8028 | 0.8271 | 0.9449 | 0.0777 | 0.5301 |
| screw | single | 1.0 | 1 | joint_best.pt | disc | 0.7799 | 0.7549 | 0.7711 | 0.9154 | 0.9089 | 0.9142 | 0.8938 | 0.1721 | 0.7301 |
| transistor | single | 1.0 | 1 | joint_best.pt | pixel | 0.4217 | 0.3962 | 0.4117 | 0.4318 | 0.3427 | 0.4054 | 0.3337 | 0.0341 | 0.0859 |
| transistor | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.8317 | 0.7392 | 0.7638 | 0.7823 | 0.6683 | 0.6830 | 0.8763 | 0.3116 | 0.5730 |
| transistor | single | 1.0 | 1 | joint_best.pt | disc | 0.6850 | 0.6142 | 0.6421 | 0.6578 | 0.5606 | 0.5868 | 0.6241 | 0.1121 | 0.5401 |
