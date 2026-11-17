# RAE-DiT IAD results (J4_latent_dtd)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.7644 | 0.9095 | 0.9085 | 0.9109 | 0.9680 | 0.9682 | 0.9805 | 0.5533 | 0.8968 |
| single | 1.0 | 1 | best | latent_cos | 0.8191 | 0.9040 | 0.9174 | 0.9226 | 0.9662 | 0.9712 | 0.9619 | 0.2980 | 0.8573 |
| single | 1.0 | 1 | best | pixel | 0.4759 | 0.6542 | 0.6559 | 0.7496 | 0.8490 | 0.8427 | 0.7612 | 0.0566 | 0.4046 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.7472 | 0.7135 | 0.7829 | 0.9158 | 0.8867 | 0.9222 | 0.6830 | 0.0968 | 0.3174 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 0.9996 | 0.9996 | 1.0000 | 0.9999 | 0.9999 | 0.9837 | 0.3968 | 0.9366 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.8868 | 0.9848 | 0.9848 | 0.9687 | 0.9957 | 0.9957 | 0.9948 | 0.8076 | 0.9661 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2046 | 0.5950 | 0.5288 | 0.5835 | 0.8112 | 0.7632 | 0.8395 | 0.0164 | 0.4918 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.6382 | 0.8084 | 0.8352 | 0.8452 | 0.9326 | 0.9424 | 0.9401 | 0.1993 | 0.7780 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.6419 | 0.8342 | 0.8323 | 0.8531 | 0.9402 | 0.9407 | 0.9661 | 0.2990 | 0.8276 |
