# RAE-DiT IAD results (ab_noise_mask_all)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5104 | 0.5748 | 0.5685 | 0.7029 | 0.7224 | 0.7171 | 0.5118 | 0.0368 | 0.1550 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4161 | 0.5060 | 0.4785 | 0.6178 | 0.6607 | 0.6458 | 0.3073 | 0.0133 | 0.0443 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4408 | 0.5739 | 0.5608 | 0.6538 | 0.7305 | 0.7338 | 0.6017 | 0.0827 | 0.3524 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5205 | 0.5233 | 0.5582 | 0.8250 | 0.8089 | 0.8381 | 0.3614 | 0.0120 | 0.0761 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2717 | 0.5562 | 0.4811 | 0.6844 | 0.7938 | 0.7708 | 0.3527 | 0.0110 | 0.0284 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.3110 | 0.5598 | 0.5249 | 0.6666 | 0.8180 | 0.7872 | 0.2406 | 0.0096 | 0.0076 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.3759 | 0.4211 | 0.3860 | 0.6850 | 0.7226 | 0.6869 | 0.5593 | 0.0079 | 0.2224 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6717 | 0.4845 | 0.4779 | 0.8260 | 0.7153 | 0.7001 | 0.5478 | 0.0073 | 0.1385 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.5656 | 0.6475 | 0.6257 | 0.7580 | 0.8349 | 0.8208 | 0.5273 | 0.0071 | 0.1235 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5021 | 0.9043 | 0.8271 | 0.6657 | 0.9497 | 0.9059 | 0.8947 | 0.3469 | 0.8538 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4146 | 0.5114 | 0.4861 | 0.5759 | 0.6568 | 0.6456 | 0.1096 | 0.0115 | 0.0000 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.4179 | 0.4521 | 0.4168 | 0.6423 | 0.6082 | 0.5976 | 0.4582 | 0.0179 | 0.0621 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.3868 | 0.6501 | 0.6399 | 0.6741 | 0.8434 | 0.8425 | 0.8536 | 0.0126 | 0.5337 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3904 | 0.5347 | 0.4687 | 0.6773 | 0.7730 | 0.7364 | 0.1531 | 0.0014 | 0.0010 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.5413 | 0.5544 | 0.5807 | 0.7847 | 0.7699 | 0.7768 | 0.5999 | 0.0031 | 0.1623 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4188 | 0.3708 | 0.3929 | 0.4192 | 0.3280 | 0.3956 | 0.3394 | 0.0340 | 0.0761 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3321 | 0.4429 | 0.4788 | 0.3257 | 0.3648 | 0.3763 | 0.3733 | 0.0351 | 0.0536 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.7163 | 0.6604 | 0.6946 | 0.6630 | 0.5812 | 0.6031 | 0.7331 | 0.1463 | 0.4192 |
