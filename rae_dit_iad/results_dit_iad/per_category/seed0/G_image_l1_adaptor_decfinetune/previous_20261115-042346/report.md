# RAE-DiT IAD results (G_image_l1_adaptor_decfinetune)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6102 | 0.6893 | 0.6857 | 0.7320 | 0.7970 | 0.7917 | 0.7370 | 0.0940 | 0.3867 |
| single | 0.1 | 1 | best | latent_cos | 0.3106 | 0.4419 | 0.4519 | 0.5660 | 0.6434 | 0.6559 | 0.5631 | 0.0399 | 0.3200 |
| single | 0.1 | 1 | best | pixel | 0.5233 | 0.6929 | 0.6812 | 0.7173 | 0.8060 | 0.7995 | 0.8756 | 0.2873 | 0.6652 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4848 | 0.4823 | 0.4920 | 0.7296 | 0.7658 | 0.7515 | 0.7371 | 0.0502 | 0.2966 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.2504 | 0.4482 | 0.4113 | 0.6282 | 0.7174 | 0.6974 | 0.3032 | 0.0103 | 0.0710 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8571 | 0.7372 | 0.7612 | 0.9525 | 0.8998 | 0.9059 | 0.6129 | 0.0190 | 0.1846 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.8329 | 0.8972 | 0.9716 | 0.9113 | 0.9506 | 0.9893 | 0.9779 | 0.3938 | 0.9362 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.3024 | 0.2615 | 0.2874 | 0.6604 | 0.6760 | 0.6887 | 0.2497 | 0.0041 | 0.0159 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.6909 | 0.7945 | 0.7778 | 0.8785 | 0.9111 | 0.9106 | 0.7216 | 0.0322 | 0.3776 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.5379 | 0.9550 | 0.9321 | 0.6957 | 0.9764 | 0.9627 | 0.9770 | 0.6344 | 0.9262 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.4211 | 0.3771 | 0.4050 | 0.5792 | 0.5970 | 0.6464 | 0.8215 | 0.0579 | 0.6469 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.4768 | 0.7711 | 0.7436 | 0.6347 | 0.8709 | 0.8534 | 0.9413 | 0.3408 | 0.8036 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0025 | 0.5796 | 0.4493 | 0.5341 | 0.8245 | 0.7499 | 0.8971 | 0.1491 | 0.6277 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.0756 | 0.6866 | 0.6983 | 0.5487 | 0.8731 | 0.8729 | 0.9387 | 0.0803 | 0.7766 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.6048 | 0.7081 | 0.6936 | 0.7884 | 0.8798 | 0.8736 | 0.7543 | 0.0060 | 0.3656 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7583 | 0.5504 | 0.5613 | 0.7157 | 0.5127 | 0.5444 | 0.7887 | 0.2090 | 0.5393 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.5033 | 0.4358 | 0.4575 | 0.4137 | 0.3534 | 0.3739 | 0.5022 | 0.0471 | 0.0893 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.4213 | 0.4358 | 0.4525 | 0.4061 | 0.4235 | 0.4149 | 0.6551 | 0.0718 | 0.2023 |
