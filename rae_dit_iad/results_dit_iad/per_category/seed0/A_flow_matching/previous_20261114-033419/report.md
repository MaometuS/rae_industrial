# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.9498 | 0.9912 | 0.9960 | 0.9861 | 0.9974 | 0.9988 | 0.9814 | 0.4666 | 0.9055 |
| single | 0.5 | 1 | best | pixel | 0.4113 | 0.4751 | 0.4856 | 0.7595 | 0.7891 | 0.7889 | 0.4118 | 0.0135 | 0.1144 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4113 | 0.4751 | 0.4856 | 0.7595 | 0.7891 | 0.7889 | 0.4118 | 0.0135 | 0.1144 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9498 | 0.9912 | 0.9960 | 0.9861 | 0.9974 | 0.9988 | 0.9814 | 0.4666 | 0.9055 |
