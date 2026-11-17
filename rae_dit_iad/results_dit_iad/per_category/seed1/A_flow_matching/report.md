# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.7622 | 0.9332 | 0.9480 | 0.8866 | 0.9769 | 0.9807 | 0.9738 | 0.4144 | 0.8945 |
| single | 0.5 | 1 | best | pixel | 0.4524 | 0.6255 | 0.6325 | 0.7547 | 0.8366 | 0.8530 | 0.6561 | 0.0203 | 0.3731 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.5702 | 0.5618 | 0.5915 | 0.8571 | 0.8195 | 0.8574 | 0.4235 | 0.0141 | 0.1268 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9169 | 0.9635 | 0.9868 | 0.9760 | 0.9885 | 0.9962 | 0.9795 | 0.5155 | 0.9138 |
| screw | single | 0.5 | 1 | screw_best.pt | pixel | 0.3345 | 0.6893 | 0.6735 | 0.6523 | 0.8537 | 0.8487 | 0.8888 | 0.0266 | 0.6194 |
| screw | single | 0.5 | 1 | screw_best.pt | latent_cos | 0.6075 | 0.9028 | 0.9092 | 0.7972 | 0.9653 | 0.9652 | 0.9681 | 0.3133 | 0.8751 |
