# RAE-DiT IAD results (G_image_l1_adaptor_decfinetune)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.7037 | 0.7052 | 0.7014 | 0.8227 | 0.8003 | 0.7994 | 0.7646 | 0.1417 | 0.4255 |
| single | 0.1 | 1 | best | latent_cos | 0.3473 | 0.4488 | 0.4393 | 0.5746 | 0.6412 | 0.6281 | 0.5126 | 0.0200 | 0.2062 |
| single | 0.1 | 1 | best | pixel | 0.5173 | 0.7117 | 0.6950 | 0.7082 | 0.8196 | 0.8066 | 0.8368 | 0.2364 | 0.6428 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4422 | 0.4535 | 0.4711 | 0.7066 | 0.7216 | 0.7335 | 0.5870 | 0.0201 | 0.1718 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.3102 | 0.3628 | 0.3166 | 0.6465 | 0.6999 | 0.6646 | 0.5280 | 0.0152 | 0.1052 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9671 | 0.8780 | 0.8764 | 0.9906 | 0.9623 | 0.9611 | 0.6257 | 0.0257 | 0.2915 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.8271 | 0.8881 | 0.9766 | 0.9213 | 0.9475 | 0.9918 | 0.9722 | 0.3704 | 0.9291 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.5489 | 0.5004 | 0.5347 | 0.7467 | 0.7405 | 0.7545 | 0.1927 | 0.0039 | 0.0064 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.7235 | 0.8722 | 0.8697 | 0.8985 | 0.9519 | 0.9530 | 0.7722 | 0.0702 | 0.5048 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.5921 | 0.9411 | 0.8982 | 0.7245 | 0.9687 | 0.9451 | 0.9686 | 0.5541 | 0.9135 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.3532 | 0.4807 | 0.4714 | 0.5672 | 0.6451 | 0.6384 | 0.4413 | 0.0187 | 0.2759 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.6782 | 0.7504 | 0.7429 | 0.7979 | 0.8543 | 0.8481 | 0.9494 | 0.4512 | 0.7420 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0121 | 0.6747 | 0.5392 | 0.5359 | 0.8438 | 0.7731 | 0.9023 | 0.0766 | 0.6767 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.3460 | 0.5464 | 0.5235 | 0.6485 | 0.7656 | 0.7365 | 0.8051 | 0.0062 | 0.4146 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.4304 | 0.5442 | 0.5395 | 0.7163 | 0.7622 | 0.7686 | 0.6895 | 0.0049 | 0.2648 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7129 | 0.6013 | 0.5900 | 0.6526 | 0.6164 | 0.5894 | 0.7537 | 0.1609 | 0.5229 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.1783 | 0.3538 | 0.3504 | 0.2641 | 0.3551 | 0.3464 | 0.5958 | 0.0557 | 0.2290 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.7192 | 0.4812 | 0.4787 | 0.7102 | 0.4709 | 0.4663 | 0.7859 | 0.1567 | 0.3246 |
