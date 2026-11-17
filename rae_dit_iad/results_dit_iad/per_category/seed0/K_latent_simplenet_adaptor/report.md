# RAE-DiT IAD results (K_latent_simplenet_adaptor)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6314 | 0.6654 | 0.6643 | 0.8150 | 0.8378 | 0.8350 | 0.6872 | 0.0336 | 0.2874 |
| single | 0.1 | 1 | best | latent_cos | 0.8200 | 0.8104 | 0.8169 | 0.9073 | 0.9154 | 0.9139 | 0.8611 | 0.1590 | 0.5793 |
| single | 0.1 | 1 | best | pixel | 0.3996 | 0.5750 | 0.5713 | 0.7175 | 0.8234 | 0.8248 | 0.5963 | 0.0110 | 0.2954 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5116 | 0.5120 | 0.5546 | 0.8234 | 0.8057 | 0.8381 | 0.3428 | 0.0114 | 0.0693 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.9647 | 0.9691 | 0.9783 | 0.9897 | 0.9911 | 0.9937 | 0.9598 | 0.2910 | 0.8107 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8909 | 0.8720 | 0.8716 | 0.9659 | 0.9621 | 0.9616 | 0.7559 | 0.0642 | 0.4715 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.2876 | 0.6380 | 0.5880 | 0.6117 | 0.8411 | 0.8115 | 0.8498 | 0.0105 | 0.5216 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.6753 | 0.6518 | 0.6555 | 0.8249 | 0.8398 | 0.8342 | 0.7624 | 0.0270 | 0.3479 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.3720 | 0.4589 | 0.4571 | 0.6642 | 0.7135 | 0.7085 | 0.6184 | 0.0029 | 0.1033 |
