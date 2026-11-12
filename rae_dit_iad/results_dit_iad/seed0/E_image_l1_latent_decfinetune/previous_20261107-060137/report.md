# RAE-DiT IAD results (E_image_l1_latent_decfinetune)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5615 | 0.4934 | 0.4955 | 0.7266 | 0.6664 | 0.6800 | 0.5032 | 0.0198 | 0.0865 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4152 | 0.4320 | 0.4123 | 0.6238 | 0.6350 | 0.6298 | 0.4472 | 0.0192 | 0.1405 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4465 | 0.5745 | 0.5581 | 0.6748 | 0.7505 | 0.7424 | 0.7254 | 0.1429 | 0.4976 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.3407 | 0.4559 | 0.4310 | 0.6931 | 0.7616 | 0.7489 | 0.3517 | 0.0186 | 0.1091 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2797 | 0.4270 | 0.3210 | 0.6782 | 0.7713 | 0.7259 | 0.6023 | 0.0229 | 0.1667 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.5317 | 0.4410 | 0.4655 | 0.8138 | 0.7444 | 0.7595 | 0.6640 | 0.0217 | 0.0872 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.7937 | 0.7527 | 0.8680 | 0.9116 | 0.8807 | 0.9501 | 0.8294 | 0.0519 | 0.6464 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6433 | 0.3450 | 0.4194 | 0.8260 | 0.6780 | 0.7122 | 0.6450 | 0.0106 | 0.2770 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4787 | 0.4946 | 0.4946 | 0.7092 | 0.7257 | 0.7183 | 0.4301 | 0.0055 | 0.0858 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5104 | 0.9229 | 0.8639 | 0.7221 | 0.9602 | 0.9272 | 0.9637 | 0.5273 | 0.9187 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3600 | 0.4904 | 0.4782 | 0.5469 | 0.6273 | 0.6139 | 0.2768 | 0.0133 | 0.0876 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7939 | 0.4193 | 0.3879 | 0.8708 | 0.6023 | 0.5922 | 0.6015 | 0.0243 | 0.0166 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0000 | 0.2125 | 0.1586 | 0.5340 | 0.6059 | 0.5836 | 0.8168 | 0.0119 | 0.4070 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4858 | 0.5128 | 0.4685 | 0.7553 | 0.7601 | 0.7623 | 0.2077 | 0.0014 | 0.0128 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.4905 | 0.5196 | 0.5210 | 0.7684 | 0.7679 | 0.7733 | 0.3417 | 0.0017 | 0.0565 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.5879 | 0.5288 | 0.4688 | 0.5132 | 0.5442 | 0.5022 | 0.6654 | 0.1047 | 0.4070 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3071 | 0.3850 | 0.3742 | 0.3125 | 0.3382 | 0.3348 | 0.5040 | 0.0479 | 0.1584 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5129 | 0.5925 | 0.6088 | 0.4705 | 0.4917 | 0.5567 | 0.4786 | 0.0457 | 0.1863 |
