# RAE-DiT IAD results (G_image_l1_adaptor_decfinetune)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5430 | 0.4765 | 0.4539 | 0.7196 | 0.6537 | 0.6361 | 0.5269 | 0.0178 | 0.0689 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3495 | 0.4384 | 0.3984 | 0.5830 | 0.6392 | 0.6252 | 0.4058 | 0.0168 | 0.0949 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4683 | 0.4911 | 0.4347 | 0.6735 | 0.7006 | 0.6767 | 0.6853 | 0.1143 | 0.4431 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.2817 | 0.3688 | 0.3443 | 0.6589 | 0.7189 | 0.7141 | 0.3407 | 0.0169 | 0.0995 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.1128 | 0.4924 | 0.3648 | 0.5835 | 0.8127 | 0.7562 | 0.5138 | 0.0153 | 0.1003 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.3451 | 0.4077 | 0.3499 | 0.7332 | 0.7191 | 0.7038 | 0.6217 | 0.0192 | 0.0358 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.2281 | 0.5054 | 0.4486 | 0.5912 | 0.7896 | 0.7390 | 0.7282 | 0.0372 | 0.4596 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3567 | 0.3225 | 0.2882 | 0.6843 | 0.6443 | 0.6255 | 0.5289 | 0.0070 | 0.1649 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4653 | 0.5455 | 0.5038 | 0.7170 | 0.7544 | 0.7044 | 0.4707 | 0.0059 | 0.0863 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.7646 | 0.8746 | 0.8021 | 0.8561 | 0.9345 | 0.8916 | 0.9399 | 0.4294 | 0.8689 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4196 | 0.4357 | 0.3968 | 0.5751 | 0.5961 | 0.5876 | 0.2483 | 0.0132 | 0.0084 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7461 | 0.3896 | 0.3954 | 0.8420 | 0.5811 | 0.5860 | 0.6619 | 0.0289 | 0.1161 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.5282 | 0.2542 | 0.1615 | 0.7859 | 0.6256 | 0.5883 | 0.8245 | 0.0124 | 0.5359 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4448 | 0.5063 | 0.5003 | 0.7173 | 0.7771 | 0.7664 | 0.2417 | 0.0015 | 0.0304 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.6313 | 0.5868 | 0.6290 | 0.8471 | 0.8108 | 0.8348 | 0.5394 | 0.0024 | 0.0640 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.5387 | 0.4525 | 0.4171 | 0.4754 | 0.4343 | 0.4503 | 0.5932 | 0.0753 | 0.2518 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4138 | 0.4350 | 0.4421 | 0.3550 | 0.3658 | 0.3905 | 0.4964 | 0.0469 | 0.1706 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5271 | 0.4529 | 0.3917 | 0.4586 | 0.4030 | 0.3513 | 0.3410 | 0.0329 | 0.0421 |
