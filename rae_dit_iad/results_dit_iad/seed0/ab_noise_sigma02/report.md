# RAE-DiT IAD results (ab_noise_sigma02)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.2 | 1 | joint_best.pt | disc | 0.5658 | 0.5650 | 0.5678 | 0.7323 | 0.7428 | 0.7438 | 0.6757 | 0.0383 | 0.1855 |
| single | 0.2 | 1 | joint_best.pt | latent_cos | 0.4951 | 0.6068 | 0.6152 | 0.6600 | 0.7233 | 0.7310 | 0.6740 | 0.0871 | 0.4405 |
| single | 0.2 | 1 | joint_best.pt | pixel | 0.4795 | 0.5841 | 0.5532 | 0.7065 | 0.7602 | 0.7427 | 0.7790 | 0.1670 | 0.5705 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.2 | 1 | joint_best.pt | pixel | 0.3387 | 0.4506 | 0.4254 | 0.7248 | 0.7933 | 0.7905 | 0.4885 | 0.0552 | 0.2324 |
| carpet | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.8242 | 0.9378 | 0.9450 | 0.9395 | 0.9808 | 0.9832 | 0.7063 | 0.2662 | 0.5488 |
| carpet | single | 0.2 | 1 | joint_best.pt | disc | 0.6122 | 0.5269 | 0.4924 | 0.8616 | 0.7802 | 0.7644 | 0.5758 | 0.0205 | 0.0904 |
| grid | single | 0.2 | 1 | joint_best.pt | pixel | 0.8814 | 0.9114 | 0.9624 | 0.9447 | 0.9621 | 0.9872 | 0.9232 | 0.0967 | 0.7896 |
| grid | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.5505 | 0.7377 | 0.7828 | 0.7828 | 0.8674 | 0.9070 | 0.7806 | 0.0822 | 0.5524 |
| grid | single | 0.2 | 1 | joint_best.pt | disc | 0.2916 | 0.5388 | 0.5539 | 0.6195 | 0.7811 | 0.7708 | 0.5574 | 0.0078 | 0.0883 |
| hazelnut | single | 0.2 | 1 | joint_best.pt | pixel | 0.3925 | 0.9075 | 0.8682 | 0.6300 | 0.9553 | 0.9328 | 0.9496 | 0.5205 | 0.8452 |
| hazelnut | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.4114 | 0.4654 | 0.4779 | 0.5782 | 0.6215 | 0.6377 | 0.7895 | 0.0487 | 0.5826 |
| hazelnut | single | 0.2 | 1 | joint_best.pt | disc | 0.7071 | 0.6071 | 0.6211 | 0.8320 | 0.7721 | 0.7847 | 0.7700 | 0.0685 | 0.1588 |
| screw | single | 0.2 | 1 | joint_best.pt | pixel | 0.0000 | 0.2476 | 0.1017 | 0.5340 | 0.6602 | 0.5598 | 0.8786 | 0.0371 | 0.5975 |
| screw | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.3277 | 0.5440 | 0.5149 | 0.6455 | 0.7710 | 0.7532 | 0.7959 | 0.0077 | 0.4968 |
| screw | single | 0.2 | 1 | joint_best.pt | disc | 0.5850 | 0.5210 | 0.5060 | 0.8017 | 0.7765 | 0.7803 | 0.7530 | 0.0055 | 0.3736 |
| transistor | single | 0.2 | 1 | joint_best.pt | pixel | 0.7850 | 0.4033 | 0.4083 | 0.6992 | 0.4301 | 0.4433 | 0.6551 | 0.1254 | 0.3876 |
| transistor | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.3617 | 0.3492 | 0.3554 | 0.3542 | 0.3760 | 0.3739 | 0.2979 | 0.0308 | 0.0217 |
| transistor | single | 0.2 | 1 | joint_best.pt | disc | 0.6333 | 0.6312 | 0.6658 | 0.5468 | 0.6042 | 0.6186 | 0.7226 | 0.0893 | 0.2164 |
