# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.9595 | 0.9976 | 0.9984 | 0.9886 | 0.9993 | 0.9995 | 0.9817 | 0.4054 | 0.9178 |
| single | 0.5 | 1 | best | pixel | 0.4868 | 0.4843 | 0.5357 | 0.8034 | 0.7753 | 0.8141 | 0.4544 | 0.0156 | 0.1306 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4868 | 0.4843 | 0.5357 | 0.8034 | 0.7753 | 0.8141 | 0.4544 | 0.0156 | 0.1306 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9595 | 0.9976 | 0.9984 | 0.9886 | 0.9993 | 0.9995 | 0.9817 | 0.4054 | 0.9178 |
