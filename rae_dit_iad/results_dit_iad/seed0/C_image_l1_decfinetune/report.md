# RAE-DiT IAD results (C_image_l1_decfinetune)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5213 | 0.5947 | 0.6059 | 0.7106 | 0.7408 | 0.7572 | 0.7210 | 0.1116 | 0.3685 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5173 | 0.5404 | 0.5406 | 0.6719 | 0.6846 | 0.6846 | 0.1892 | 0.0179 | 0.0313 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4777 | 0.5833 | 0.6011 | 0.6981 | 0.7479 | 0.7682 | 0.7947 | 0.2211 | 0.5466 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5060 | 0.5345 | 0.5494 | 0.7766 | 0.8150 | 0.8194 | 0.5620 | 0.1343 | 0.3073 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7195 | 0.8415 | 0.8411 | 0.8944 | 0.9525 | 0.9507 | 0.1430 | 0.0088 | 0.0032 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.1224 | 0.2970 | 0.2913 | 0.5899 | 0.6744 | 0.6728 | 0.7928 | 0.0590 | 0.4443 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.7444 | 0.7260 | 0.7945 | 0.8748 | 0.8683 | 0.9180 | 0.8298 | 0.0600 | 0.5825 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5338 | 0.7226 | 0.7310 | 0.7506 | 0.8723 | 0.8828 | 0.0331 | 0.0035 | 0.0000 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.6550 | 0.8145 | 0.8129 | 0.7994 | 0.9046 | 0.9044 | 0.6946 | 0.0173 | 0.3830 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5439 | 0.9121 | 0.9179 | 0.7411 | 0.9565 | 0.9566 | 0.9835 | 0.7350 | 0.8796 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2761 | 0.2332 | 0.2171 | 0.5449 | 0.5007 | 0.4911 | 0.0965 | 0.0112 | 0.0016 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7546 | 0.7982 | 0.7968 | 0.8521 | 0.8885 | 0.8874 | 0.9370 | 0.2740 | 0.6633 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0000 | 0.2974 | 0.1863 | 0.5340 | 0.6854 | 0.6235 | 0.8930 | 0.0621 | 0.6503 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6061 | 0.5889 | 0.6102 | 0.7963 | 0.7855 | 0.7945 | 0.0842 | 0.0013 | 0.0000 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.5126 | 0.5581 | 0.5794 | 0.7659 | 0.7674 | 0.7759 | 0.5373 | 0.0066 | 0.0813 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.5942 | 0.4467 | 0.5575 | 0.5640 | 0.4143 | 0.5238 | 0.7052 | 0.1143 | 0.3133 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4508 | 0.3158 | 0.3038 | 0.3733 | 0.3121 | 0.3036 | 0.5891 | 0.0647 | 0.1516 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5621 | 0.5058 | 0.5492 | 0.5455 | 0.4693 | 0.5454 | 0.6430 | 0.2011 | 0.2707 |
