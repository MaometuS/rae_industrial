# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.6127 | 0.8795 | 0.8738 | 0.8261 | 0.9493 | 0.9460 | 0.8981 | 0.4643 | 0.8185 |
| single | 1.0 | 1 | best | latent_cos | 0.8820 | 0.9055 | 0.9021 | 0.9421 | 0.9568 | 0.9555 | 0.9303 | 0.3532 | 0.7769 |
| single | 1.0 | 1 | best | pixel | 0.4709 | 0.6647 | 0.6404 | 0.6582 | 0.8014 | 0.7893 | 0.7129 | 0.1717 | 0.4823 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6493 | 0.5979 | 0.6533 | 0.8706 | 0.8366 | 0.8696 | 0.6369 | 0.0947 | 0.2911 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9988 | 0.9988 | 0.9988 | 0.9996 | 0.9996 | 0.9996 | 0.9863 | 0.4385 | 0.9524 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2849 | 0.8648 | 0.8600 | 0.7378 | 0.9432 | 0.9433 | 0.9908 | 0.7359 | 0.9398 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4879 | 0.6291 | 0.5380 | 0.6980 | 0.8473 | 0.7759 | 0.7539 | 0.1143 | 0.5012 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9885 | 0.3166 | 0.9579 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.4194 | 0.9808 | 0.9816 | 0.7767 | 0.9930 | 0.9936 | 0.9855 | 0.2435 | 0.9408 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.4843 | 0.9661 | 0.9225 | 0.6462 | 0.9831 | 0.9595 | 0.8857 | 0.5352 | 0.8714 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.9939 | 0.9925 | 0.9925 | 0.9964 | 0.9956 | 0.9956 | 0.9795 | 0.5043 | 0.8688 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.9936 | 0.9879 | 0.9900 | 0.9960 | 0.9927 | 0.9939 | 0.9935 | 0.8474 | 0.9616 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2318 | 0.6522 | 0.5915 | 0.5924 | 0.8546 | 0.8162 | 0.8728 | 0.0535 | 0.5637 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.4567 | 0.5929 | 0.5737 | 0.7629 | 0.8588 | 0.8498 | 0.8264 | 0.1903 | 0.4776 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.4183 | 0.6159 | 0.5866 | 0.6803 | 0.8678 | 0.8475 | 0.8395 | 0.2513 | 0.5719 |
| transistor | single | 1.0 | 1 | transistor_best.pt | pixel | 0.5012 | 0.4783 | 0.4967 | 0.4836 | 0.4853 | 0.5254 | 0.4153 | 0.0608 | 0.1838 |
| transistor | single | 1.0 | 1 | transistor_best.pt | latent_cos | 0.9608 | 0.9433 | 0.9454 | 0.9516 | 0.9300 | 0.9326 | 0.8708 | 0.3166 | 0.6278 |
| transistor | single | 1.0 | 1 | transistor_best.pt | disc | 0.9471 | 0.9483 | 0.9508 | 0.9399 | 0.9500 | 0.9518 | 0.6814 | 0.2432 | 0.6781 |
