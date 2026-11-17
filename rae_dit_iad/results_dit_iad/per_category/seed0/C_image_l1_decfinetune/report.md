# RAE-DiT IAD results (C_image_l1_decfinetune)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6263 | 0.6795 | 0.6671 | 0.7431 | 0.7908 | 0.7865 | 0.7580 | 0.1504 | 0.3658 |
| single | 0.1 | 1 | best | latent_cos | 0.2869 | 0.4379 | 0.4269 | 0.5473 | 0.6541 | 0.6462 | 0.4951 | 0.0233 | 0.2445 |
| single | 0.1 | 1 | best | pixel | 0.5106 | 0.7224 | 0.6947 | 0.7064 | 0.8261 | 0.8041 | 0.8364 | 0.2264 | 0.6421 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4326 | 0.4599 | 0.4607 | 0.7098 | 0.7565 | 0.7290 | 0.5964 | 0.0220 | 0.1843 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0919 | 0.2155 | 0.1673 | 0.5752 | 0.6396 | 0.6274 | 0.2339 | 0.0095 | 0.0355 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9029 | 0.8660 | 0.8535 | 0.9700 | 0.9606 | 0.9576 | 0.6439 | 0.0236 | 0.2272 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.8421 | 0.8914 | 0.9783 | 0.9260 | 0.9497 | 0.9921 | 0.9687 | 0.3218 | 0.9194 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.3492 | 0.5505 | 0.5614 | 0.6671 | 0.8161 | 0.8194 | 0.3299 | 0.0048 | 0.0686 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5221 | 0.8446 | 0.8262 | 0.7254 | 0.9362 | 0.9290 | 0.7945 | 0.0794 | 0.5468 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.5554 | 0.9550 | 0.9071 | 0.7124 | 0.9752 | 0.9495 | 0.9694 | 0.5601 | 0.9167 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.2839 | 0.4507 | 0.4443 | 0.5203 | 0.6141 | 0.6113 | 0.3421 | 0.0148 | 0.1431 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.6118 | 0.6143 | 0.5879 | 0.7423 | 0.7840 | 0.7666 | 0.9375 | 0.4478 | 0.5648 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0141 | 0.7040 | 0.5468 | 0.5359 | 0.8482 | 0.7701 | 0.9005 | 0.0734 | 0.6725 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.3851 | 0.5356 | 0.5433 | 0.6597 | 0.7611 | 0.7636 | 0.8824 | 0.0101 | 0.6032 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5626 | 0.5347 | 0.5460 | 0.7678 | 0.7711 | 0.7779 | 0.6528 | 0.0039 | 0.1825 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7088 | 0.6017 | 0.5804 | 0.6479 | 0.6009 | 0.5799 | 0.7471 | 0.1549 | 0.5178 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.3246 | 0.4371 | 0.4179 | 0.3142 | 0.4397 | 0.4094 | 0.6874 | 0.0771 | 0.3723 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5321 | 0.5379 | 0.5217 | 0.5096 | 0.5020 | 0.5016 | 0.7614 | 0.1971 | 0.3075 |
