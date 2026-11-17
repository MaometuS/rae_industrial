# RAE-DiT IAD results (ab_noise_mask_all)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.4718 | 0.6690 | 0.6780 | 0.6615 | 0.8029 | 0.8027 | 0.3050 | 0.0430 | 0.1014 |
| single | 0.1 | 1 | best | latent_cos | 0.2991 | 0.4103 | 0.3682 | 0.5862 | 0.6231 | 0.6100 | 0.3544 | 0.0140 | 0.1671 |
| single | 0.1 | 1 | best | pixel | 0.4812 | 0.7172 | 0.6803 | 0.6863 | 0.8294 | 0.8074 | 0.7855 | 0.2262 | 0.6089 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.3788 | 0.4807 | 0.4579 | 0.7162 | 0.7774 | 0.7609 | 0.4139 | 0.0192 | 0.1257 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0470 | 0.2584 | 0.1738 | 0.5624 | 0.6714 | 0.6158 | 0.0280 | 0.0081 | 0.0002 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.1950 | 0.7010 | 0.7313 | 0.6112 | 0.8900 | 0.9062 | 0.1943 | 0.0092 | 0.0030 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.6633 | 0.8697 | 0.8713 | 0.7684 | 0.9369 | 0.9472 | 0.9437 | 0.3106 | 0.8686 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.3542 | 0.3918 | 0.3668 | 0.6717 | 0.6906 | 0.6810 | 0.4111 | 0.0062 | 0.1201 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5614 | 0.6449 | 0.6274 | 0.7862 | 0.8346 | 0.8199 | 0.2261 | 0.0040 | 0.0271 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.6368 | 0.9507 | 0.9154 | 0.7717 | 0.9716 | 0.9519 | 0.9719 | 0.5743 | 0.9252 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.3114 | 0.5161 | 0.5361 | 0.5692 | 0.6192 | 0.6731 | 0.1206 | 0.0115 | 0.0000 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.6564 | 0.6750 | 0.7129 | 0.7756 | 0.8008 | 0.8209 | 0.0552 | 0.0109 | 0.0166 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0127 | 0.6729 | 0.5339 | 0.5356 | 0.8608 | 0.7856 | 0.8820 | 0.0750 | 0.6092 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.6952 | 0.5856 | 0.5790 | 0.8809 | 0.8171 | 0.8129 | 0.8945 | 0.0124 | 0.6499 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.4489 | 0.5328 | 0.5245 | 0.6901 | 0.7323 | 0.7236 | 0.4006 | 0.0020 | 0.1171 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7146 | 0.6121 | 0.6229 | 0.6393 | 0.6001 | 0.5914 | 0.7161 | 0.1520 | 0.5162 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.0875 | 0.2996 | 0.1854 | 0.2467 | 0.3174 | 0.2674 | 0.3179 | 0.0317 | 0.0655 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.4971 | 0.7910 | 0.7937 | 0.4445 | 0.7570 | 0.7430 | 0.6487 | 0.1890 | 0.3432 |
