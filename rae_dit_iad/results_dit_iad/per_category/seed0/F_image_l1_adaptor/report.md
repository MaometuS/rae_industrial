# RAE-DiT IAD results (F_image_l1_adaptor)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6509 | 0.6690 | 0.6594 | 0.7901 | 0.7797 | 0.7740 | 0.7751 | 0.1078 | 0.4466 |
| single | 0.1 | 1 | best | latent_cos | 0.2383 | 0.3929 | 0.3740 | 0.5535 | 0.6253 | 0.6269 | 0.3052 | 0.0159 | 0.1680 |
| single | 0.1 | 1 | best | pixel | 0.4974 | 0.6904 | 0.6702 | 0.7040 | 0.8086 | 0.8001 | 0.7690 | 0.2003 | 0.6028 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.3447 | 0.4535 | 0.4290 | 0.6995 | 0.7736 | 0.7518 | 0.3631 | 0.0162 | 0.1161 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0959 | 0.3202 | 0.3046 | 0.5829 | 0.6927 | 0.6901 | 0.0294 | 0.0081 | 0.0007 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8768 | 0.7632 | 0.7548 | 0.9634 | 0.9261 | 0.9226 | 0.7321 | 0.0443 | 0.3992 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.7561 | 0.8371 | 0.9165 | 0.8750 | 0.9194 | 0.9653 | 0.9441 | 0.2307 | 0.8743 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.1487 | 0.3383 | 0.3016 | 0.5616 | 0.6677 | 0.6575 | 0.0157 | 0.0035 | 0.0000 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.6658 | 0.8563 | 0.8404 | 0.8697 | 0.9438 | 0.9384 | 0.7330 | 0.0491 | 0.4335 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.6464 | 0.9550 | 0.9257 | 0.7775 | 0.9736 | 0.9567 | 0.9712 | 0.5705 | 0.9226 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.3525 | 0.5454 | 0.5857 | 0.6162 | 0.6755 | 0.7406 | 0.1187 | 0.0114 | 0.0000 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.5846 | 0.8071 | 0.7700 | 0.7046 | 0.8876 | 0.8688 | 0.9434 | 0.3205 | 0.8030 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0146 | 0.6331 | 0.5296 | 0.5358 | 0.8355 | 0.7782 | 0.8836 | 0.0581 | 0.6171 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.4817 | 0.5247 | 0.4972 | 0.7518 | 0.7893 | 0.7746 | 0.9326 | 0.0181 | 0.7721 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.4347 | 0.5306 | 0.5251 | 0.7260 | 0.7661 | 0.7635 | 0.6894 | 0.0056 | 0.2700 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7254 | 0.5733 | 0.5500 | 0.6321 | 0.5409 | 0.5487 | 0.6832 | 0.1261 | 0.4837 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.1125 | 0.2358 | 0.1808 | 0.2548 | 0.3014 | 0.2716 | 0.4294 | 0.0386 | 0.0672 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.6925 | 0.3875 | 0.4067 | 0.6870 | 0.3750 | 0.3770 | 0.7779 | 0.1195 | 0.3272 |
