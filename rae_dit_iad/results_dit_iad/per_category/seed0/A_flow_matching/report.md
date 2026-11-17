# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.7601 | 0.9244 | 0.9417 | 0.8843 | 0.9703 | 0.9781 | 0.9739 | 0.3941 | 0.8880 |
| single | 0.5 | 1 | best | pixel | 0.4651 | 0.6389 | 0.6354 | 0.7534 | 0.8546 | 0.8538 | 0.6587 | 0.0217 | 0.3832 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.5758 | 0.5867 | 0.6224 | 0.8527 | 0.8408 | 0.8605 | 0.4201 | 0.0142 | 0.1205 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9105 | 0.9567 | 0.9771 | 0.9739 | 0.9869 | 0.9934 | 0.9774 | 0.5014 | 0.8980 |
| screw | single | 0.5 | 1 | screw_best.pt | pixel | 0.3544 | 0.6911 | 0.6485 | 0.6540 | 0.8684 | 0.8470 | 0.8973 | 0.0293 | 0.6460 |
| screw | single | 0.5 | 1 | screw_best.pt | latent_cos | 0.6098 | 0.8922 | 0.9063 | 0.7946 | 0.9538 | 0.9628 | 0.9703 | 0.2869 | 0.8780 |
