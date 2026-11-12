# RAE-DiT IAD results (C_image_l1_decfinetune)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5739 | 0.4945 | 0.5019 | 0.7329 | 0.6676 | 0.6823 | 0.4808 | 0.0190 | 0.0825 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4037 | 0.4397 | 0.4107 | 0.6110 | 0.6359 | 0.6190 | 0.4144 | 0.0184 | 0.1106 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4035 | 0.5923 | 0.5550 | 0.6478 | 0.7598 | 0.7393 | 0.7256 | 0.1440 | 0.5010 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.3648 | 0.4747 | 0.4535 | 0.7047 | 0.7689 | 0.7599 | 0.3628 | 0.0197 | 0.1174 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2171 | 0.5457 | 0.4169 | 0.6500 | 0.8290 | 0.7723 | 0.5480 | 0.0168 | 0.1206 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.5706 | 0.4470 | 0.4755 | 0.8346 | 0.7473 | 0.7631 | 0.6663 | 0.0219 | 0.0897 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.7485 | 0.7953 | 0.8622 | 0.8861 | 0.9100 | 0.9465 | 0.8478 | 0.0605 | 0.6841 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5840 | 0.3292 | 0.3375 | 0.7684 | 0.6421 | 0.6507 | 0.6247 | 0.0092 | 0.2203 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4737 | 0.4971 | 0.4937 | 0.7070 | 0.7273 | 0.7136 | 0.4234 | 0.0054 | 0.0855 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.2218 | 0.9214 | 0.8721 | 0.5262 | 0.9595 | 0.9318 | 0.9522 | 0.5198 | 0.9035 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3996 | 0.4379 | 0.4100 | 0.5643 | 0.6089 | 0.5828 | 0.2078 | 0.0125 | 0.0150 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7886 | 0.4111 | 0.3789 | 0.8672 | 0.5999 | 0.5866 | 0.5759 | 0.0227 | 0.0103 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0000 | 0.2455 | 0.1097 | 0.5340 | 0.6215 | 0.5627 | 0.8070 | 0.0119 | 0.3784 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4226 | 0.4698 | 0.4243 | 0.7271 | 0.7540 | 0.7273 | 0.1544 | 0.0014 | 0.0003 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.5128 | 0.5411 | 0.5577 | 0.7785 | 0.7831 | 0.8008 | 0.2728 | 0.0016 | 0.0487 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.6825 | 0.5246 | 0.4775 | 0.5879 | 0.5391 | 0.4959 | 0.6581 | 0.1081 | 0.4216 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3954 | 0.4158 | 0.4650 | 0.3455 | 0.3454 | 0.3618 | 0.5371 | 0.0522 | 0.1969 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5238 | 0.5763 | 0.6038 | 0.4770 | 0.4806 | 0.5477 | 0.4658 | 0.0433 | 0.1781 |
