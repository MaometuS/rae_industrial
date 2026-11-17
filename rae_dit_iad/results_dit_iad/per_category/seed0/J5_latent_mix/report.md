# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.5585 | 0.8968 | 0.9000 | 0.8354 | 0.9621 | 0.9631 | 0.9365 | 0.4498 | 0.8165 |
| single | 1.0 | 1 | best | latent_cos | 0.8733 | 0.9554 | 0.9518 | 0.9482 | 0.9850 | 0.9837 | 0.9800 | 0.3234 | 0.9044 |
| single | 1.0 | 1 | best | pixel | 0.4632 | 0.6531 | 0.6604 | 0.7427 | 0.8575 | 0.8575 | 0.7678 | 0.0632 | 0.4362 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6810 | 0.6685 | 0.7368 | 0.8891 | 0.8691 | 0.9046 | 0.6702 | 0.1045 | 0.3141 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9791 | 0.3744 | 0.9103 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2949 | 0.9912 | 0.9876 | 0.7480 | 0.9972 | 0.9961 | 0.9886 | 0.7347 | 0.9539 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2453 | 0.6376 | 0.5841 | 0.5964 | 0.8459 | 0.8104 | 0.8654 | 0.0218 | 0.5583 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7467 | 0.9108 | 0.9037 | 0.8963 | 0.9700 | 0.9674 | 0.9809 | 0.2724 | 0.8985 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.8221 | 0.8024 | 0.8125 | 0.9228 | 0.9269 | 0.9301 | 0.8844 | 0.1648 | 0.6792 |
