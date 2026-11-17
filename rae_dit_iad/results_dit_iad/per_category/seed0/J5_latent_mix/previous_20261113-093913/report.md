# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.4005 | 0.9009 | 0.9057 | 0.7970 | 0.9538 | 0.9581 | 0.9939 | 0.7730 | 0.9609 |
| single | 1.0 | 1 | best | latent_cos | 0.9988 | 0.9988 | 0.9988 | 0.9996 | 0.9996 | 0.9996 | 0.9832 | 0.3966 | 0.9421 |
| single | 1.0 | 1 | best | pixel | 0.6902 | 0.6192 | 0.6970 | 0.8810 | 0.8476 | 0.8840 | 0.6280 | 0.0955 | 0.2779 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6902 | 0.6192 | 0.6970 | 0.8810 | 0.8476 | 0.8840 | 0.6280 | 0.0955 | 0.2779 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9988 | 0.9988 | 0.9988 | 0.9996 | 0.9996 | 0.9996 | 0.9832 | 0.3966 | 0.9421 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.4005 | 0.9009 | 0.9057 | 0.7970 | 0.9538 | 0.9581 | 0.9939 | 0.7730 | 0.9609 |
