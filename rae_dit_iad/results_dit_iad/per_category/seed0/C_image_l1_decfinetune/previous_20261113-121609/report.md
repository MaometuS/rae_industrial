# RAE-DiT IAD results (C_image_l1_decfinetune)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.5909 | 0.5401 | 0.5329 | 0.7252 | 0.6962 | 0.6970 | 0.7449 | 0.0720 | 0.3278 |
| single | 0.1 | 1 | best | latent_cos | 0.3464 | 0.3950 | 0.3575 | 0.6051 | 0.6184 | 0.5952 | 0.4174 | 0.0216 | 0.1135 |
| single | 0.1 | 1 | best | pixel | 0.5361 | 0.7431 | 0.7190 | 0.7284 | 0.8339 | 0.8160 | 0.8466 | 0.2469 | 0.6374 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4077 | 0.4900 | 0.4771 | 0.6843 | 0.7396 | 0.7192 | 0.6194 | 0.0218 | 0.1587 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0221 | 0.1437 | 0.0654 | 0.5666 | 0.6280 | 0.5792 | 0.0642 | 0.0082 | 0.0070 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9149 | 0.6601 | 0.6509 | 0.9743 | 0.8882 | 0.8855 | 0.7625 | 0.0634 | 0.4373 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.8981 | 0.9624 | 0.9908 | 0.9506 | 0.9862 | 0.9969 | 0.9650 | 0.2857 | 0.8993 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.4194 | 0.6792 | 0.6190 | 0.6667 | 0.8668 | 0.8344 | 0.3737 | 0.0050 | 0.0451 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.3826 | 0.5113 | 0.4937 | 0.6439 | 0.7433 | 0.7452 | 0.7184 | 0.0328 | 0.3545 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.6500 | 0.9811 | 0.9786 | 0.8128 | 0.9899 | 0.9875 | 0.9785 | 0.6955 | 0.9178 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.3496 | 0.3654 | 0.3196 | 0.5946 | 0.5644 | 0.5348 | 0.4836 | 0.0200 | 0.2037 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.5846 | 0.4793 | 0.4750 | 0.7466 | 0.6290 | 0.6235 | 0.8625 | 0.0890 | 0.4299 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0156 | 0.6760 | 0.5386 | 0.5356 | 0.8520 | 0.7768 | 0.9041 | 0.0571 | 0.6844 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.2888 | 0.3892 | 0.3804 | 0.6254 | 0.6992 | 0.6849 | 0.5132 | 0.0023 | 0.0695 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5466 | 0.5188 | 0.5339 | 0.7575 | 0.7341 | 0.7547 | 0.6221 | 0.0030 | 0.1250 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7092 | 0.6062 | 0.6100 | 0.6588 | 0.6015 | 0.5997 | 0.7662 | 0.1742 | 0.5266 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.6521 | 0.3975 | 0.4029 | 0.5722 | 0.3334 | 0.3429 | 0.6522 | 0.0724 | 0.2421 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5258 | 0.5312 | 0.5108 | 0.5039 | 0.4863 | 0.4760 | 0.7589 | 0.1718 | 0.2925 |
