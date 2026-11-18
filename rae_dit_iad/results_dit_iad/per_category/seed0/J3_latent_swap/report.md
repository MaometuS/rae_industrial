# RAE-DiT IAD results (J3_latent_swap)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.8024 | 0.8124 | 0.8061 | 0.9180 | 0.9240 | 0.9199 | 0.8475 | 0.2176 | 0.6286 |
| single | 1.0 | 1 | best | latent_cos | 0.8422 | 0.9008 | 0.9066 | 0.9273 | 0.9595 | 0.9633 | 0.9759 | 0.3711 | 0.9013 |
| single | 1.0 | 1 | best | pixel | 0.3726 | 0.5705 | 0.5570 | 0.7055 | 0.8205 | 0.8163 | 0.6014 | 0.0112 | 0.3006 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.4944 | 0.5008 | 0.5369 | 0.8130 | 0.7971 | 0.8283 | 0.3502 | 0.0116 | 0.0716 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9791 | 0.9803 | 0.9968 | 0.9942 | 0.9944 | 0.9990 | 0.9879 | 0.5405 | 0.9622 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9703 | 0.9555 | 0.9607 | 0.9914 | 0.9856 | 0.9870 | 0.9597 | 0.4239 | 0.8403 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2509 | 0.6403 | 0.5772 | 0.5979 | 0.8439 | 0.8044 | 0.8526 | 0.0109 | 0.5295 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7053 | 0.8213 | 0.8164 | 0.8605 | 0.9246 | 0.9277 | 0.9639 | 0.2017 | 0.8405 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.6346 | 0.6694 | 0.6516 | 0.8445 | 0.8624 | 0.8529 | 0.7353 | 0.0113 | 0.4168 |
